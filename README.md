# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Problem Statement

Regression problems aim to predict continuous numerical values from given input features. Traditional regression methods (like Linear Regression) assume a linear relationship between variables. However, many real-world datasets exhibit non‑linear relationships. Neural Networks can learn complex and non‑linear mappings between inputs and outputs, making them highly suitable for regression tasks.

## Neural Network Model

<img width="917" height="806" alt="image" src="https://github.com/user-attachments/assets/e4076740-1f06-4403-bb1c-f12ea3f0a06c" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:Mahasri D
### Register Number:212224220058
```python
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 16)   # single hidden layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)            # output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = NeuralNet(X_train.shape[1])
print(model)




# Initialize the Model, Loss Function, and Optimizer
model = NeuralNet(X_train.shape[1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return losses
  
```


## Dataset Information

<img width="487" height="603" alt="image" src="https://github.com/user-attachments/assets/81b0f365-5c37-4426-aa26-8524847a800a" />

## OUTPUT

### Training Loss Vs Iteration Plot

<img width="803" height="578" alt="image" src="https://github.com/user-attachments/assets/83d3ef5b-3042-4556-8469-056ad1212016" />


### New Sample Data Prediction

<img width="672" height="343" alt="image" src="https://github.com/user-attachments/assets/49419016-8de7-4e39-b5bf-db50bf5a2cdc" />

## RESULT

Thus the program To develop a neural network regression model for the given dataset has been done successfully.
