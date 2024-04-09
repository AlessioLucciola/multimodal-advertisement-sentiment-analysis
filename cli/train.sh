python -m  cli.hparams_tuning --batch-size=128 --architecture=resnet34 --epochs=20 --dataset-limit=0.5 --normalize --apply_transformations --balance_dataset --use_default_split

python -m  cli.hparams_tuning --batch-size=128 --architecture=resnet50 --epochs=20 --dataset-limit=0.5 --normalize --apply_transformations --balance_dataset --use_default_split

python -m  cli.hparams_tuning --batch-size=128 --architecture=resnet101 --epochs=20 --dataset-limit=0.5 --normalize --apply_transformations --balance_dataset --use_default_split

python -m  cli.hparams_tuning --batch-size=128 --architecture=densenet121 --epochs=20 --dataset-limit=0.5 --normalize --apply_transformations --balance_dataset --use_default_split

python -m  cli.hparams_tuning --batch-size=128 --architecture=custom-cnn --epochs=20 --dataset-limit=0.5 --normalize --apply_transformations --balance_dataset --use_default_split

python -m  cli.hparams_tuning --batch-size=128 --architecture=vit-pretrained --dataset-limit=0.5 --epochs=20 --normalize --apply_transformations --balance_dataset --use_default_split
