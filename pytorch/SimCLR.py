class SimCLR(nn.Module):
    def __init__(self, x, lr=0.01, d_model=512, gamma_temperature=0.5):
        self.x = x
        super(SimCLR, self).__init__()
        self.gamma_temperature = gamma_temperature

        self.f = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.f.fc = nn.Identity()
        self.f.eval()
        
        dummy_input = torch.randn_like(self.x[0].unsqueeze(0))
        
        output = self.f(dummy_input)
        self.f.train()

        self.d_model = d_model
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(output.size(1),d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model,d_model)
        )

        self.device = "cpu"
        self.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)


    def random_crop(self):
        crop = torchvision.transforms.RandomResizedCrop((224,224),scale=(0.08,1),ratio=(3/4, 4/3),antialias=True)
        flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
    
        augment = torchvision.transforms.Compose([
            crop,
            flip
        ])
        return augment 
        
    def color_distortions(self,s=1.0):#strength
        factor_1 = 0.8*s
        factor_2 = 0.2*s
        
        color_jitter = torchvision.transforms.ColorJitter(brightness=factor_1, contrast=factor_1, saturation=factor_1, hue=factor_2)
        rnd_apply_jitter = torchvision.transforms.RandomApply([color_jitter],p=0.8)
    
        rnd_gray = torchvision.transforms.RandomGrayscale(p=0.2)
        color_distort = torchvision.transforms.Compose([
                rnd_apply_jitter,
                rnd_gray
        ])
    
        return color_distort

    def gaussian_blur(self,x):
        _, height, width = x.size()
        kernel_size = np.ceil(height/width * 0.1)
    
        gauss_blur = torchvision.transforms.GaussianBlur(kernel_size,sigma=(0.1,2.0))
        blur_apply = torchvision.transforms.RandomApply([gauss_blur],p=0.5)
        return blur_apply

    def get_augmentation(self):
        aug = np.random.choice([self.gaussian_blur,self.color_distortions,self.random_crop],size=2, replace=False)
        return aug

    def similarity(self, x):
        sim_dict = {}
        for i in range(len(x)):
            for j in range(len(x)):
                sim_ij = self.calculate_similarity(x[i], x[j])
                sim_dict[(i, j)] = sim_ij[0]

        return sim_dict

    def calculate_similarity(self,x_1,x_2):
        if len(x_1.shape) == 1:
            x_1 = x_1.unsqueeze(0)
        if len(x_2.shape) == 1:
            x_2 = x_2.unsqueeze(0)
        return torch.mm(x_1,x_2.t())/ (torch.norm(x_1) * torch.norm(x_2))

    
    def NT_Xent_loss(self, s, i, j):
        denominator = 0
        numerator = torch.exp(s[(i, j)] / self.gamma_temperature)
    
        for (k, l), sim_value in s.items():
            if k == i and l != i:
                denominator += torch.exp(sim_value / self.gamma_temperature)
    
        return -torch.log(numerator / denominator)

    def loss(self, s, batch_size):
        Total = 0
        for k in range(batch_size):
            Total += self.NT_Xent_loss(s, 2*k, 2*k + 1) + self.NT_Xent_loss(s, 2*k + 1, 2*k)

        return 1/batch_size * Total
    
    def forward(self, x):
        self.train()
        z = torch.zeros((x.size(0)*2,self.d_model))
        x = x.to(self.device)

        batch_size = x.size(0)
        
        for k in range(len(x)):
            augmentation = self.get_augmentation()
            aug_1 = augmentation[0](x[k]) if augmentation[0].__name__ == 'gaussian_blur' else augmentation[0]()
            aug_2 = augmentation[1](x[k]) if augmentation[1].__name__ == 'gaussian_blur' else augmentation[1]()

            z[2*k] = self.MLP(self.f (aug_1(x[k]).unsqueeze(0)))
            z[2*k + 1] = self.MLP(self.f( aug_2(x[k]).unsqueeze(0)))

        s = self.similarity(z)
        loss = self.loss(s, batch_size)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        print(loss)