import torch
from torchmetrics.regression import MeanAbsolutePercentageError
from model import GIN, hidden_channels, out_channels
from processing import encode_date

def run_model(checkpoint_filepath, input_data, day):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GIN(hidden_channels, out_channels)
    checkpoint = torch.load(checkpoint_filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    input_data = input_data.to(device)
    date_feat = encode_date(day).to(device).unsqueeze(0)

    if not hasattr(input_data, 'batch') or input_data.batch is None:
        batch = torch.zeros(input_data.x.size(0), dtype=torch.long, device=device)
    else:
        batch = input_data.batch

    with torch.no_grad():
        predictions = model(input_data.x, input_data.edge_index, batch, date_feat)

    # Reverses normalization of tensors
    return torch.expm1(predictions)

# Calculates mean absolute percentage error for validation
def calculate_validation_error(model_prediction, target):
    mape = MeanAbsolutePercentageError()
    return mape(model_prediction.cpu(), target).item() * 100

# Calculates average validation error across a set of input data and target lists
def calculate_set_validation_error(checkpoint_filepath, input_data_list, target_list, date_list):
    total_error = 0
    number_of_inputs = len(input_data_list)
    
    if len(target_list) != number_of_inputs:
        print("Number of input data in validation set does not match number of targets.")
        return
    
    for i in range(number_of_inputs):
        model_prediction = run_model(checkpoint_filepath, input_data_list[i], date_list[i])
        total_error += calculate_validation_error(model_prediction, target_list[i])

    return total_error/number_of_inputs

checkpoint_filepath = "data/checkpoints/checkpoint_epoch49.pt"
input_data = torch.load("data/processed/2019-02-24_2019-03-10_Ang Thong.pt", weights_only=False)
day = 62
prediction = run_model(checkpoint_filepath, input_data, day)
print("Predicted tensor: ", prediction)
target_tensor = torch.tensor([0,0,192033,0,0,0,0,46853,0,0,0,82300,0,28447], dtype=torch.float)
print("Target tensor: ", target_tensor)
print("Validation error: ", calculate_validation_error(prediction, target_tensor))