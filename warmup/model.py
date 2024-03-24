import torch

class SimpleModel(torch.nn.Module):
    
    def __init__(self, img_size = (28,28), label_num = 10):
        super().__init__()
        self.label_num = label_num
        
        ## Modify the code below ##

        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(img_size[0]*img_size[1], label_num),
            torch.nn.Softmax(dim=1)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
