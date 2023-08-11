import argparse
from utils import *

if __name__ == "__main__":
    '''
    this is training wraining ....
    '''
    parser = argparse.ArgumentParser(
        description="Select and train model on your dataset"
    )

    parser.add_argument(
        "data_directory", help="data directory for training model")
    parser.add_argument("--save_dir", help="directory to save the checkpoints")
    parser.add_argument("--arch", help="pretrained Model Architecture to use")
    parser.add_argument("--learning_rate",
                        help="learning rate to use for training")
    parser.add_argument("--epochs", help="number of epochs for training")
    parser.add_argument(
        "--hidden_units", help="number of hidden units to use for building classifier")
    parser.add_argument("--gpu", action="store_true", help="train on GPU")
    argums = parser.parse_args()

    device = None
    if argums.gpu and torch.cuda.is_available():
        print("Training on gpu")
        device = torch.device("cuda")
        # device = torch_directml.device()

    elif not torch.cuda.is_available():
        print("GPU not avilable so training on cpu")
        device = torch.device("cpu")

    else:
        print("Training on cpu")
        device = torch.device("cpu")

    arch = (argums.arch if argums.arch else 'efficientnet_b0')
    n_hidden = (int(argums.hidden_units) if argums.hidden_units else 0)
    lr = (float(argums.learning_rate) if argums.learning_rate else 0.001)
    epochs = (int(argums.epochs) if argums.epochs else 1)
    data_dir = argums.data_directory
    save_dir = (argums.save_dir if argums.save_dir else "Models")

    model = build_model(arch, n_hidden)
    data_loaders, class_to_idx = load_and_process_data(data_dir)
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    trainloader = data_loaders["training"]
    validloader = data_loaders["validation"]
    testloader = data_loaders["testing"]

    optimizer = training(model, optimizer=optimizer, trainloader=trainloader,
                         validloader=validloader, epochs=epochs, device=device)
    model.class_to_idx = class_to_idx
    save_model(model, optimizer, save_dir, arch, epochs)
