from pathlib import Path
import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data
import datahandler
from model import createDeepLabv3
from trainer import train_model, eval_test_dataset_new


torch.cuda.empty_cache()
#print(torch.cuda.device(0))
# python eval_test_dataset.py --data-directory CADIS --batch-size 1

@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=False,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=25,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
def main(data_directory, exp_directory, epochs, batch_size):
    #model = torch.load('./CADIS/dark_weights/CADIS_weights_temp_epoch_5_25.pt')
    model = torch.load('./CADIS/light_weights/CADIS_weights_temp_epoch_1_25.pt')
    #model = torch.load('./CADIS/light_weights/CADIS_weights_temp_epoch_2_25.pt')
    # Set the model to evaluate mode
    model.eval()
    data_directory = Path(data_directory)

    # Create the dataloader
    dataloaders_test_dataset = datahandler.get_dataloader_single_folder_eval(
        data_directory, batch_size=batch_size)

    _ = eval_test_dataset_new(model,
                    dataloaders_test_dataset)


if __name__ == "__main__":
    main()
