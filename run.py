import torch
from torchmetrics.regression import MeanAbsolutePercentageError
from torcheval.metrics.functional import r2_score
from model import GAT, hidden_channels, out_channels, heads
from processing import encode_date

def run_model(checkpoint_filepath, input_data, day):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GAT(hidden_channels, out_channels, heads)
    checkpoint = torch.load(checkpoint_filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    feature_mean = checkpoint['feature_mean']
    feature_std = checkpoint['feature_std']
    
    print("feature_mean:", feature_mean)
    print("feature_std:", feature_std)
    print("feature_std max:", feature_std.max().item())
    print("feature_std min:", feature_std.min().item())

    assert torch.isfinite(input_data.x).all(), "non-finite values in input_data.x before normalization"
    assert torch.isfinite(feature_mean).all(), "non-finite feature_mean in checkpoint"
    assert torch.isfinite(feature_std).all(), "non-finite feature_std in checkpoint"

    input_data = input_data.to(device)
    input_data.x = (input_data.x - feature_mean) / (feature_std + 1e-8)

    assert torch.isfinite(input_data.x).all(), "non-finite values in input_data.x after normalization"

    date_feat = encode_date(day).to(device).unsqueeze(0)

    if not hasattr(input_data, 'batch') or input_data.batch is None:
        batch = torch.zeros(input_data.x.size(0), dtype=torch.long, device=device)
    else:
        batch = input_data.batch

    with torch.no_grad():
        predictions = model(input_data.x, input_data.edge_index, batch, date_feat)

    return torch.expm1(predictions)

# Calculates mean absolute percentage error for validation
def calculate_validation_error(model_prediction, target):
    mape = MeanAbsolutePercentageError()
    return mape(model_prediction.cpu().squeeze(), target).item() * 100

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

checkpoint_filepath = "data/checkpoints/checkpoint_best.pt"
input_data = torch.load("data/processed/2019-02-19_2019-03-05_Si Sa Ket.pt", weights_only=False)
day = 57
prediction = run_model(checkpoint_filepath, input_data, day).squeeze()
target_tensor = torch.tensor([398,31474,512583,104915,1890,9946,146881,61805,56181,3183,7328,219678,14024,37524], dtype=torch.float)
r_squared = r2_score(prediction.cpu(), target_tensor)
print("Predicted tensor: ", prediction)
print("Target tensor: ", target_tensor)
print("Validation error: ", calculate_validation_error(prediction, target_tensor))
print("R^2 score: ", r_squared)