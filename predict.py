from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.")

    parser.add_argument('image_path', help = "path to the image file")
    parser.add_argument('checkpoint',help='directory to models checkpoint, for e.g. Models/efficientnet')
    parser.add_argument('--top_k', help="Topk most likey classes according ton models prediction")
    parser.add_argument('--category_name', help="filepath that contains mapping to classes for each labels")
    parser.add_argument('--gpu', help="If you want to use gpu for inference", action="store_true")
    args = parser.parse_args()

    device = None
    if args.gpu and torch.cuda.is_available():
        print("Prediction on gpu")
        device = torch.device("cuda")
        # device = torch_directml.device()

    elif not torch.cuda.is_available():
        print("GPU not avilable so Predicting on cpu")
        device = torch.device("cpu")

    else:
        print("Inference on cpu")
        device = torch.device("cpu")

    topk = (int(args.top_k) if args.top_k else 5)
    category_name = (args.category_name if args.category_name else "cat_to_name.json")

    __, model, _ = load_model(path=args.checkpoint, device=device)

    model.eval()
    prob, classes = predict(image_path=args.image_path, model= model,device=device, topk=topk)
    
    with open(category_name, 'r') as f:
        cat_to_label = json.load(f)

    class_labels = [cat_to_label[c] for c in classes]

    print(f"Predicted flower {class_labels[0]} with probabilty of {prob[0]:.4f}")
    
    for c,p in zip(class_labels, prob):
        print(f"{c} : {p}")

