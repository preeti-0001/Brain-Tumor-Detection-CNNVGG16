from cnn.custom_cnn import build_cnn
from cnn.vgg16 import build_vgg16
from data.data_fetch import fetch_data
from data.data_preprocess import create_generators
from data.data_split import split_data
from evaluate.evaluate import evaluate_model, plot_model_comparison
from utilities.plot_history import plot_history

from testing.predict import predict_sample
from training.train_models import train_cnn_model, train_vgg16_model



def main():
    fetch_data()
    split_data()
    train_gen, val_gen, test_gen = create_generators()
    
    cnn_model = build_cnn()
    cnn_model.summary()

    
    vgg_model, vgg_base = build_vgg16()
    vgg_model.summary()
    vgg_h1, vgg_h2 = train_vgg16_model(vgg_model, vgg_base, train_gen, val_gen)
    cnn_history = train_cnn_model(cnn_model, train_gen, val_gen)

    
    plot_history(cnn_history, 'Custom CNN',        '#3498db')
    plot_history(vgg_h1,      'VGG16 Phase 1',     '#e67e22')
    plot_history(vgg_h2,      'VGG16 Fine Tune',   '#9b59b6')

    
    cnn_acc = evaluate_model(cnn_model, test_gen, 'Custom CNN')
    vgg_acc = evaluate_model(vgg_model, test_gen, 'VGG16 Transfer Learning')
    plot_model_comparison(cnn_acc, vgg_acc)

    predict_sample(vgg_model, cnn_model)


if __name__ == "__main__":
    main()