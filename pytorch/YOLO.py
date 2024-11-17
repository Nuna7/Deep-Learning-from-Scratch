import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.leaky_relu(self.norm(x))
        return x

class YOLO(nn.Module):
    def __init__(self, num_class, S, B, conf_threshold, iou_threshold, lambda_coord=0.5, lambda_noobj=5, lr=1e-4):
        super(YOLO, self).__init__()
        self.num_class = num_class
        self.S = S
        self.B = B
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lr = lr

        self.conv_layer = self.create_conv()

        self.final_layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(1024 * 7 * 7, 4096),
                nn.Dropout(0.5),
                nn.LeakyReLU(0.1),
                nn.Linear(4096, self.S * self.S * (5 * self.B + self.num_class))
        )
 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
    
        for param in self.parameters():
            param.requires_grad = True

    def create_conv(self):
        conv_layer = nn.Sequential(
            ConvNet(3, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvNet(64, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvNet(192, 128, kernel_size=1),
            ConvNet(128, 256, kernel_size=3),
            ConvNet(256, 256, kernel_size=1),
            ConvNet(256, 512, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvNet(512, 256, kernel_size=1),
            ConvNet(256, 512, kernel_size=3),
            ConvNet(512, 256, kernel_size=1),
            ConvNet(256, 512, kernel_size=3),
            ConvNet(512, 256, kernel_size=1),
            ConvNet(256, 512, kernel_size=3),
            ConvNet(512, 256, kernel_size=1),
            ConvNet(256, 512, kernel_size=3),
            ConvNet(512, 512, kernel_size=1),
            ConvNet(512, 1024, kernel_size=3),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvNet(1024, 512, kernel_size=1),
            ConvNet(512, 1024, kernel_size=3),
            ConvNet(1024, 512, kernel_size=1),
            ConvNet(512, 1024, kernel_size=3),
            ConvNet(1024, 1024, kernel_size=3),
            ConvNet(1024, 1024, kernel_size=3, stride=2),
            ConvNet(1024, 1024, kernel_size=3),
            ConvNet(1024, 1024, kernel_size=3),
            
        )
        return conv_layer

    def forward(self, x):
        n, c, h, w = x.size()
        x = self.conv_layer(x)
        out = self.final_layer(x)
        out = out.view(n, self.S, self.S, (5 * self.B + self.num_class))
        for b in range(self.B):
            out[:, :, :, b*5+4] = torch.sigmoid(out[:, :, :, b*5+4])
        out[:, :, :, self.B*5 : ] = self.softmax(out[:, :, :, self.B * 5 : ])
        return out

    def get_xyxy(self, bboxes):
        x, y, w, h = bboxes
        return [x - w/2,y - h/2,x + w/2,y + h/2]

    def iou(self, box1, box2):
        box1_xyxy = self.get_xyxy(box1[-4:])
        box2_xyxy = self.get_xyxy(box2[-4:])
        
        x1 = torch.max(box1_xyxy[0], box2_xyxy[0])
        y1 = torch.max(box1_xyxy[1], box2_xyxy[1])
        x2 = torch.min(box1_xyxy[2], box2_xyxy[2])
        y2 = torch.min(box1_xyxy[3], box2_xyxy[3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        box1_area = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        box2_area = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        
        union = box1_area + box2_area - intersection
        return intersection / (union + 1e-6)
    
    def check_conf(self, bbox, class_probs):
        obj_conf = bbox[4]
        class_conf = obj_conf * class_probs
        class_pred = torch.argmax(class_conf)
        max_conf = class_conf[class_pred]
        if max_conf >= self.conf_threshold:
            return True, class_pred.item(), obj_conf.item()
        return False, None, None

    
    def loss(self, predictions, labels):
        """
        predictions: batch_size, S, S, (B * 5 + num_class)
        labels: batch_size, S, S, (B * 5 + num_class)
        """
        batch_size = predictions.size(0)
        
        # Extract coordinates and class predictions
        pred_xy = predictions[..., :2 * self.B].view(batch_size, self.S, self.S, self.B, 2)
        pred_wh = predictions[..., 2 * self.B:4 * self.B].view(batch_size, self.S, self.S, self.B, 2)
        
        # Extract object confidence scores
        pred_conf = predictions[..., [4 * i for i in range(self.B)]].view(batch_size, self.S, self.S, self.B)
        
        # Extract class predictions
        pred_class = predictions[..., 5 * self.B:].view(batch_size, self.S, self.S, self.num_class)
        
        # Extract ground truth
        true_xy = labels[..., :2 * self.B].view(batch_size, self.S, self.S, self.B, 2)
        true_wh = labels[..., 2 * self.B:4 * self.B].view(batch_size, self.S, self.S, self.B, 2)
        
        # Extract object confidence scores from labels
        true_conf = labels[..., [4 * i for i in range(self.B)]].view(batch_size, self.S, self.S, self.B)
        
        # Extract class labels
        true_class = labels[..., 5 * self.B:].view(batch_size, self.S, self.S, self.num_class)
        
        # Object mask
        obj_mask = true_conf  # Shape: [batch_size, S, S, B]
        noobj_mask = 1 - obj_mask
        
        # Coordinate loss
        xy_loss = obj_mask.unsqueeze(-1) * (pred_xy - true_xy)**2
        wh_loss = obj_mask.unsqueeze(-1) * (torch.sqrt(pred_wh) - torch.sqrt(true_wh))**2
        coord_loss = self.lambda_coord * torch.sum(xy_loss + wh_loss)
        
        # Confidence loss
        conf_loss_obj = obj_mask * (pred_conf - true_conf)**2
        conf_loss_noobj = noobj_mask * (pred_conf - true_conf)**2
        conf_loss = torch.sum(conf_loss_obj + self.lambda_noobj * conf_loss_noobj)
        
        # Class loss
        class_mask = obj_mask.any(dim=-1) 
        class_loss = class_mask.unsqueeze(-1) * (pred_class - true_class)**2
        class_loss = torch.sum(class_loss)

        coord_loss = coord_loss.float()
        conf_loss = conf_loss.float()
        class_loss = class_loss.float()
    
        # Total loss
        total_loss = coord_loss + conf_loss + class_loss
        
        return total_loss / batch_size

    #Inference
    @torch.no_grad()
    def nms(self, predictions):
        batch_size = predictions.shape[0]
        all_boxes = torch.zeros((batch_size, self.S, self.S, self.B * 5 + self.num_class))

        for batch in range(batch_size):
            all_class = set()
            boxes_in_image = []

            for row in range(self.S):
                for column in range(self.S):
                    for b in range(self.B):
                        bbox = predictions[batch, row, column, b*5:(b*5+5)]
                        class_probs = predictions[batch, row, column, self.B*5:]

                        is_valid, class_pred, obj_conf = self.check_conf(bbox, class_probs)
                        if is_valid:
                            x, y, w, h = bbox[:4]
                            x = (column + x.item()) / self.S
                            y = (row + y.item()) / self.S
                            w = w.item() / self.S
                            h = h.item() / self.S

                            boxes_in_image.append([class_pred, x, y, w, h, obj_conf] + class_probs.tolist() + [row, column, b])
                            all_class.add(class_pred)

            for cls in all_class:
                cls_boxes = [box for box in boxes_in_image if box[0] == cls]
                cls_boxes.sort(key=lambda x: x[5], reverse=True)

                while cls_boxes:
                    chosen_box = cls_boxes.pop(0)
                    row, column, b = chosen_box[-3:]
                    all_boxes[batch, row, column, b*5:(b*5+5)] = torch.tensor([chosen_box[1], chosen_box[2], chosen_box[3], chosen_box[4], chosen_box[5]])
                    all_boxes[batch, row, column, self.B*5:] = torch.tensor(chosen_box[6:-3])

                    cls_boxes = [box for box in cls_boxes if self.iou(torch.tensor(chosen_box[1:5]), torch.tensor(box[1:5])) < self.iou_threshold]

        return all_boxes

    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
        

    def train(self, EPOCHS, data_loader):
        device = self.device()
        self.to(device)  
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr) 
    
        for epoch in range(EPOCHS):
            self.train()  # Use 'self' instead of 'model'
            train_loss = 0
            for batch_images, batch_labels in data_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            
                optimizer.zero_grad()
                predictions = self(batch_images)  # Use 'self' instead of 'model'
                loss = self.loss(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_loss = train_loss / len(data_loader)
            print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_loss:.4f}")

        return self
        
    def predict(self, x):
        x = self.forward(x)
        x = self.nms(x)
        return x