from open_clip import create_model_and_transforms
from template_tokenizer import template_tokenize
import torchvision.transforms as T
import torch
import torch.nn.functional as F
from utils import read_avi

# You'll need to log in to the HuggingFace hub CLI to download the models
# You can do this with the terminal command "huggingface-cli login"
# You'll be asked to paste your HuggingFace API token, which you can find at https://huggingface.co/settings/token

# Use EchoCLIP-R for retrieval-based tasks where you want to find
# the similarity between two echos, like in patient identification or
# echo report retrieval. It has a longer context window because it
# uses the template tokenizer, which we found increases its retrieval
# performance but decreases its performance on other zero-shot tasks.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "hf-hub:mkaichristensen/echo-clip-r"
video_path = "example_video.avi"

echo_clip_r, _, preprocess_val = create_model_and_transforms(
    model_path, precision="float16", device=device
)



# We'll load a sample echo video and preprocess its frames.
test_video = read_avi(
    video_path,
    (224, 224),
)
test_video = torch.stack(
    [preprocess_val(T.ToPILImage()(frame)) for frame in test_video], dim=0
)
test_video = test_video.to(device)
test_video = test_video.to(torch.float16)

# Be sure to normalize the CLIP embeddings after calculating them to make
# cosine similarity between embeddings easier to calculate.
echo_clip_r = echo_clip_r.to(torch.float16)
test_video = test_video.to(torch.float16)
test_video_embedding = F.normalize(echo_clip_r.encode_image(test_video), dim=-1)

import torch.nn as nn


class TransformerEmbeddingGenerator(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerEmbeddingGenerator, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_layers
        )
        self.output_proj = nn.Linear(d_model, input_dim)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        output = self.output_proj(x[:, -1, :])
        return F.normalize(output, dim=-1)

class EmbeddingPredictor:
    def __init__(self, model_path, device):
        self.device = device
        self.echo_clip_r, _, self.preprocess_val = create_model_and_transforms(
            model_path, precision="float16", device=device
        )
        self.echo_clip_r = self.echo_clip_r.to(torch.float16)
        
        input_dim = 512  #512 dim shape
        d_model = 512
        nhead = 8
        num_layers = 6
        dim_feedforward = 1024
        self.autoregressive_model = TransformerEmbeddingGenerator(
            input_dim, d_model, nhead, num_layers, dim_feedforward
        ).to(device)
        
        self.optimizer = optim.Adam(self.autoregressive_model.parameters(), lr=1e-4)
        self.loss_fn = nn.CosineSimilarity(dim=1)
    
    def preprocess_video(self, video_path):
        video = read_avi(video_path, (224, 224))
        video = torch.stack(
            [self.preprocess_val(T.ToPILImage()(frame)) for frame in video], dim=0
        )
        video = video.to(self.device).to(torch.float16)
        return torch.tensor(video, dtype=torch.float16)
    
    def compute_embeddings(self, video):
        with torch.no_grad():
            embeddings = F.normalize(self.echo_clip_r.encode_image(video), dim=-1)
        return embeddings
    
    def train(self, video_path, sequence_length, num_epochs):
        video = self.preprocess_video(video_path)
        embeddings = self.compute_embeddings(video)
        
        for epoch in range(num_epochs):
            total_loss = 0
            for i in range(len(embeddings) - sequence_length):
                sequence = embeddings[i:i+sequence_length]
                target = embeddings[i+sequence_length]
                
                self.optimizer.zero_grad()
                prediction = self.autoregressive_model(torch.tensor(sequence.unsqueeze(0), dtype=torch.float16))
                loss = 1 - self.loss_fn(prediction, target.unsqueeze(0)).mean()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
    
    def predict_next_embedding(self, initial_sequence):
        with torch.no_grad():
            prediction = self.autoregressive_model(initial_sequence.unsqueeze(0))
        return prediction.squeeze(0)
    

predictor = EmbeddingPredictor(model_path, device)
predictor.train(video_path, sequence_length=10, num_epochs=100)

video = predictor.preprocess_video(video_path)
embeddings = predictor.compute_embeddings(video)
initial_sequence = embeddings[:10] 
next_embedding = predictor.predict_next_embedding(initial_sequence)

print("Predicted next embedding:", next_embedding)