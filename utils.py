from requirements import *

device = torch_directml.device()
print(device)


def build_model(arch, n_hidden):
    """
    load a pretrained model from torchvision and rebuild classifier

    Parameters:
    arch(str) : model architecture to use, e.g. "vgg16"
    n_hidden(int) : number of hidden layers in classifier

    """
    m = models
    with open("./model_to_out.json") as mods:
        model_archs = json.loads(mods.read())
        mods.close()

    classifier_in_features = model_archs.get(arch, '0')
    if classifier_in_features == '0':
        print("Model info not there... Use another model")
        exit()

    else:
        print(
            f"Found model {arch} with no of features in input of classifier as {classifier_in_features}")

    model = eval("m." + arch + "(pretrained = False)")
    # print(model)

    # freezing model parameters
    for params in model.parameters():
        params.requires_grad = False

    # updating classifier for pretrained model

    if n_hidden == 0:
        n_hidden = (classifier_in_features + 102) // 2

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(classifier_in_features, n_hidden)),
        ('reluc1', nn.ReLU()),
        ('dpC2', nn.Dropout(0.4)),
        ('fc2', nn.Linear(n_hidden, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    return model


# Training the model
def training(model, trainloader, validloader, optimizer=None, epochs=1,  device=torch.device("cpu")):
    """
    train the model

    train model either from scratch or resume training, for given datasets and print training loss, validation loss and validation accuracy at each epoch. Also print batch training loss after every 50 batches.

    Parameters:
    model : model to train
    optimizer : Adam, SGD or any other optimizer
    trainloader : dataloader for training dataset
    validloader : dataloader for validation dataset
    epochs : number of epochs, default 1

    Returns:
    optimizer : optimizer with current state, to use for further training

    """

    if optimizer == None:
        optimizer = optim.Adam(model.classifier.params(), lr=0.001)

    criterion = nn.NLLLoss()
    # start training
    # for each epoch do validation check
    for params in model.classifier.parameters():
        params.requires_grad = True
    print("Training Started on :", device)
    sys.stdout.flush()
    model.to(device)
    for epoch in range(epochs):
        running_train_loss = 0
        step = 0
        for image, label in trainloader:
            image, label = image.to(device), label.to(device)
            model.train()
            optimizer.zero_grad()
            logp = model.forward(image)
            loss = criterion(logp, label)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            step += 1
            if step % 50 == 0:
                print(
                    f"Training- Epoch : {epoch}  Batch: {step}   Loss: {loss.item()}")

        # validation check
        running_valid_loss = 0
        valid_batch_acc = []
        for image, label in validloader:
            image, label = image.to(device), label.to(device)
            model.eval()
            with torch.no_grad():
                logp = model.forward(image)
                loss = criterion(logp, label)
                running_valid_loss += loss.item()
                # accuracy calc
                logp = torch.exp(logp)
                _, flower_pred = logp.topk(1, dim=1)
                equal = label == flower_pred.view(*label.shape)
                acc = torch.mean(equal.float()) * 100
                valid_batch_acc.append(acc.item())

        print(f"Epoch: {epoch}\t\t\tTraining Loss: {running_train_loss/len(trainloader)}"
              f"\nValidaion Loss: {running_valid_loss/len(validloader)}"
              f"\t\t\tValidation Accuracy: {round(sum(valid_batch_acc)/len(valid_batch_acc), 2)}%")
    return optimizer


def load_and_process_data(data_dir):
    """
    preprocess load and return training, validation and testing dataloader

    Parameters:
    data_dir(string) : directory at which data is present

    Return:
    dataloader(dict) : dataloader dictionary "training", "validation", "testing" as key and their corresponding datasets as value.
    class_to_idx
    """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    data_transforms = {
        "training": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),

        "validation": transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),

        "testing": transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)

        ])}

    #  Load the datasets with ImageFolder
    image_datasets = {}
    image_datasets["training"] = datasets.ImageFolder(
        train_dir, transform=data_transforms["training"])
    image_datasets["validation"] = datasets.ImageFolder(
        valid_dir, transform=data_transforms["validation"])
    image_datasets["testing"] = datasets.ImageFolder(
        test_dir, transform=data_transforms["testing"])

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {}
    for name, dataset in image_datasets.items():
        dataloaders[name] = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=(name == "training"))

    return dataloaders, image_datasets["training"].class_to_idx


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # reading from internet
    img = None
    try:
        img = Image.open(image).convert("RGB")
    except:
        try:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            http = urllib3.PoolManager()
            resp = http.request("GET", image)
            image_file = io.BytesIO(resp.data)
            img = Image.open(image_file).convert("RGB")
        except:
            raise FileExistsError("Sorry I can't get ", image_file)

    resize = transforms.Resize(256)
    cent_crop = transforms.CenterCrop(224)
    img = resize(img)
    img = cent_crop(img)
    np_image = np.array(img)
    np_image = np_image/255

    # normalise
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    tens_trans = transforms.ToTensor()
    tensor_image = tens_trans(img)
    tensor_image = tensor_image.type(torch.FloatTensor)
    norm = transforms.Normalize(mean, std)
    tensor_image = norm(tensor_image)
    return tensor_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    if title:
        ax.set_title(title)
    plt.show()
    return ax


def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Parameters:
    image_path(str) : path to local machine image or web-link
    model : which model you want to use for inference
    topk(int) : topk most likely classes which will be returned

    Return:
    (probabilty , classes)
    '''


    model.to(device)
    image = process_image(image_path)
    image = image.to(device)
    model.eval()
    with torch.no_grad():
        batched_single_image = image.unsqueeze(0)
        x = model(batched_single_image)
        x = torch.exp(x)
        prob, classes = x.topk(topk, dim=1)
        prob, classes = prob.to(torch.device(
            "cpu")), classes.to(torch.device("cpu"))
        prob, classes = prob.squeeze().tolist(), classes.squeeze().tolist()
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        if topk==1:
            return [prob], [idx_to_class[classes]]
        class_idx = [idx_to_class[c] for c in classes]
        return prob, class_idx


def save_model(model, optimizer, save_dir, arch, epochs=1):
    """
    Save model as checkpoint

    Parameters:
    model: model to save
    save_dir(str) : directory where to save model
    """

    checkpoint = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "classifier": model.classifier,
        "class_to_idx" : model.class_to_idx,
        "epochs": epochs
    }
    model_dir = save_dir + '/' + arch
    torch.save(checkpoint, model_dir)
    print("Model saved at: ", model_dir)


def load_model(path, device):
    '''
    args:
    path: path to the checkpoint
    return:
    tuple : (epoch(int), model, optimizer)
    '''
    # gety model name
    model_name = re.split(r'/|\\', path)[-1]

    if model_name.__contains__(".pth"):
        model_name = model_name.replace(".pth", '')

    # load model
    model = eval("models." + model_name + "(pretrained = False)")

    if not path.__contains__(".pth"):
        path = path + ".pth"

    loaded_checkpoint = torch.load(path, map_location=device)
    epoch = loaded_checkpoint["epochs"]

    model.classifier = loaded_checkpoint["classifier"]
    model.load_state_dict(loaded_checkpoint["model_state"])
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(loaded_checkpoint["optim_state"])
    model.class_to_idx = loaded_checkpoint["class_to_idx"]
    model.eval()

    return epoch, model, optimizer
