from torchsummary import summary

def show_model_summary(model, input_size=(3, 28, 28)):
    summary(model, input_size=input_size)
