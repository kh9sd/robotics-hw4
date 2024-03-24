import torch

class SimpleModel(torch.nn.Module):
    
    def __init__(self, img_size = (28,28), label_num = 10):
        super().__init__()
        self.label_num = label_num
        
        ## Modify the code below ##
        """
        Suggested Architecture: 
        Conv2d(32, kernel=3, stride=1, padding=1) → 
        ReLU → 
        Pool(kernel=2, stride=2) → 
        Conv2d(64, kernel=3, stride=1, padding=1) → 
        ReLU → 
        Pool(kernel=2, stride=2) → 
        Flatten → 
        FC →
        Softmax
        """
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Flatten(),
            torch.nn.Linear(img_size[0]*img_size[1]*4, label_num),
            torch.nn.Softmax(dim=1)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
        # for layer in self.layers:
        #     print(layer)
        #     x = layer(x)

        #     print(x.size())
        # return x
