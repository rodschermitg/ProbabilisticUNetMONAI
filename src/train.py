import json
import os
import time

import monai
import sklearn.model_selection
import torch

import config
import models
import utils


monai.utils.set_determinism(config.RANDOM_STATE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = 4 if device.type == "cuda" else 0
pin_memory = True if device.type == "cuda" else False
print(f"Using {device} device")

data_path = os.path.join(config.data_dir, config.DATA_FILENAME)
with open(data_path, "r") as data_file:
    data = json.load(data_file)
# entire dataset is first stored into CacheDataset and later extracted into
# separate Subsets during cross validation
dataset = monai.data.CacheDataset(
    data=data["train"],
    transform=config.base_transforms,
    num_workers=num_workers
)

model = models.ProbabilisticSegmentationNet(**config.MODEL_KWARGS).to(device)
model.init_weights(torch.nn.init.kaiming_uniform_, 0)
model.init_bias(torch.nn.init.constant_, 0)
discretize = monai.transforms.AsDiscrete(argmax=True, to_onehot=2)

optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    min_lr=1e-5,
    verbose=True
)

loss_fn_nll = torch.nn.NLLLoss(reduction="mean")
loss_fn_kl = torch.distributions.kl_divergence
dice_fn = monai.metrics.DiceMetric(include_background=False)
confusion_matrix_fn = monai.metrics.ConfusionMatrixMetric(
    include_background=False,
    metric_name=("precision", "recall")
)

k_fold = sklearn.model_selection.KFold(
    n_splits=config.FOLDS,
    shuffle=True,
    random_state=config.RANDOM_STATE
)
fold_indices = k_fold.split(dataset)

best_fold = -1
best_epoch = -1
best_dice = -1

train_logs = {
    "total_train_time": -1,
    "fold_indices": {f"fold{i}": {} for i in range(config.FOLDS)},
    "mean_train_loss_nll": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_train_loss_kl": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_loss_nll": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_dice": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_precision": {f"fold{i}": [] for i in range(config.FOLDS)},
    "mean_val_recall": {f"fold{i}": [] for i in range(config.FOLDS)},
    "best_fold": best_fold,
    "best_epoch": best_epoch,
    "best_dice": best_dice
}

print("Training network")
train_start_time = time.perf_counter()

for fold, (train_indices, val_indices) in enumerate(fold_indices):
    train_logs["fold_indices"][f"fold{fold}"] = {
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist()
    }

    train_data = torch.utils.data.Subset(dataset, train_indices)
    val_data = torch.utils.data.Subset(dataset, val_indices)

    train_data = monai.data.Dataset(train_data, config.train_transforms)
    val_data = monai.data.CacheDataset(
        val_data,
        config.eval_transforms,
        num_workers=num_workers,
        progress=False
    )

    train_dataloader = monai.data.DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dataloader = monai.data.DataLoader(
        dataset=val_data,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    for epoch in range(config.EPOCHS):
        model.train()

        epoch_train_loss_nll = 0
        epoch_train_loss_kl = 0

        for iter, train_batch in enumerate(train_dataloader):
            train_images = train_batch["images"].to(device)
            train_label = train_batch["label"].to(device)

            with torch.cuda.amp.autocast():
                train_preds = model(
                    train_images,
                    train_label,
                    make_onehot=False
                )
                train_loss_nll = loss_fn_nll(
                    train_preds,
                    torch.argmax(train_label, dim=1)  # decode one-hot labels
                )
                train_loss_kl = loss_fn_kl(model.posterior, model.prior).mean()
                train_loss = train_loss_nll + config.KL_WEIGHT*train_loss_kl

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss_nll += train_loss_nll.item()
            epoch_train_loss_kl += train_loss_kl.item()

            if iter == 0:
                print("")
            if (iter + 1) % config.DISPLAY_INTERVAL == 0:
                print(
                    f"Fold [{fold+1:1}/{config.FOLDS}], "
                    f"Epoch [{epoch+1:3}/{config.EPOCHS}], "
                    f"Iter [{iter+1:2}/{len(train_dataloader)}], "
                    f"NLL loss: {train_loss_nll.item():.4f}, "
                    f"KL loss: {train_loss_kl.item():.4f}"
                )

        mean_train_loss_nll = epoch_train_loss_nll / len(train_dataloader)
        mean_train_loss_kl = epoch_train_loss_kl / len(train_dataloader)
        train_logs["mean_train_loss_nll"][f"fold{fold}"].append(
            mean_train_loss_nll
        )
        train_logs["mean_train_loss_kl"][f"fold{fold}"].append(
            mean_train_loss_kl
        )
        print(f"Mean train NLL loss: {mean_train_loss_nll:.4f}")
        print(f"Mean train KL loss: {mean_train_loss_kl:.4f}")

        if (epoch + 1) % config.VAL_INTERVAL == 0:
            model.eval()

            epoch_val_loss_nll = 0

            with torch.no_grad():
                for val_batch in val_dataloader:
                    val_images = val_batch["images"].to(device)
                    val_label = val_batch["label"].to(device)
                    with torch.cuda.amp.autocast():
                        val_preds = monai.inferers.sliding_window_inference(
                            inputs=val_images,
                            roi_size=config.PATCH_SIZE,
                            sw_batch_size=config.BATCH_SIZE,
                            predictor=model
                        )

                    epoch_val_loss_nll += loss_fn_nll(
                        val_preds,
                        torch.argmax(val_label, dim=1)
                    ).item()

                    # store discretized batches in lists for metric functions
                    val_preds = [
                        discretize(val_pred)
                        for val_pred in monai.data.decollate_batch(val_preds)
                    ]
                    val_label = monai.data.decollate_batch(val_label)
                    # metric results are stored internally
                    dice_fn(val_preds, val_label)
                    confusion_matrix_fn(val_preds, val_label)

            mean_val_loss_nll = epoch_val_loss_nll / len(val_dataloader)
            mean_val_dice = dice_fn.aggregate().item()
            mean_val_precision = confusion_matrix_fn.aggregate()[0].item()
            mean_val_recall = confusion_matrix_fn.aggregate()[1].item()

            dice_fn.reset()
            confusion_matrix_fn.reset()

            train_logs["mean_val_loss_nll"][f"fold{fold}"].append(
                mean_val_loss_nll
            )
            train_logs["mean_val_dice"][f"fold{fold}"].append(mean_val_dice)
            train_logs["mean_val_precision"][f"fold{fold}"].append(
                mean_val_precision
            )
            train_logs["mean_val_recall"][f"fold{fold}"].append(
                mean_val_recall
            )

            print(f"Mean val NLL loss: {mean_val_loss_nll:.4f}")
            print(f"Mean val dice: {mean_val_dice:.4f}")
            print(f"Mean val precision: {mean_val_precision:.4f}")
            print(f"Mean val recall: {mean_val_recall:.4f}")

            if mean_val_dice > best_dice:
                print("New best dice, serializing model to disk")

                best_fold = fold
                best_epoch = epoch
                best_dice = mean_val_dice

                train_logs["best_fold"] = best_fold
                train_logs["best_epoch"] = best_epoch
                train_logs["best_dice"] = best_dice

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "fold": best_fold,
                        "epoch": best_epoch
                    },
                    os.path.join(config.model_dir, f"{config.MODEL_NAME}.tar")
                )
            else:
                print(
                    f"Current best dice: {best_dice:.4f} at fold "
                    f"{best_fold+1}, epoch {best_epoch+1}"
                )

            scheduler.step(mean_val_loss_nll)

            utils.create_log_plots(
                y_list=[
                    # concat all lists for each fold into a single list
                    utils.concat_logs(train_logs["mean_train_loss_nll"]),
                    utils.concat_logs(train_logs["mean_val_loss_nll"])
                ],
                fold=fold,
                epoch=epoch,
                labels=["train NLL loss", "val NLL loss"],
                output_path=config.nll_loss_plot_path,
                title="mean training/validation NLL loss per epoch",
                y_label="Loss"
            )
            utils.create_log_plots(
                y_list=[utils.concat_logs(train_logs["mean_train_loss_kl"])],
                fold=fold,
                epoch=epoch,
                labels=["train KL loss"],
                output_path=config.kl_loss_plot_path,
                title="mean training KL loss per epoch",
                y_label="Loss"
            )
            utils.create_log_plots(
                y_list=[
                    utils.concat_logs(train_logs["mean_val_dice"]),
                    utils.concat_logs(train_logs["mean_val_precision"]),
                    utils.concat_logs(train_logs["mean_val_recall"])
                ],
                fold=fold,
                epoch=epoch,
                labels=["val dice", "val precision", "val recall"],
                output_path=config.val_metric_plot_path,
                best_fold=best_fold,
                best_epoch=best_epoch,
                best_dice=best_dice,
                title="mean validation metrics per epoch",
                y_label="Metric"
            )

            with open(config.train_logs_path, "w") as train_logs_file:
                json.dump(train_logs, train_logs_file, indent=4)

    if config.SAVE_MODEL_EACH_FOLD:
        print(f"Fold {fold+1} completed, serializing model to disk")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "fold": fold,
                "epoch": epoch
            },
            os.path.join(
                config.model_dir, f"{config.MODEL_NAME}_fold{fold}.tar"
            )
        )

train_end_time = time.perf_counter()
total_train_time = train_end_time - train_start_time
train_logs["total_train_time"] = total_train_time
print(f"\nTraining finished after {total_train_time:.0f}s")

with open(config.train_logs_path, "w") as train_logs_file:
    json.dump(train_logs, train_logs_file, indent=4)


# TODO
# - add hyperparameter optimization
# - review lr scheduling
