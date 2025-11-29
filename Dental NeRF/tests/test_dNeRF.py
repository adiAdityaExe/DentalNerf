def test_forward_pass():
    import torch
    from dNeRF import YourModelClass  # Replace with the actual model class name

    model = YourModelClass()  # Initialize your model
    input_tensor = torch.randn(1, 60)  # Adjust the shape as needed
    output = model(input_tensor)

    assert output is not None
    assert output[0].shape[0] == 65536  # Adjust based on expected output shape
    assert output[1].shape[0] == 128  # Adjust based on expected output shape

test_forward_pass()