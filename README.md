# Neural Network Movie Rating Prediction
## Project Overview
A deep learning-based movie recommendation system built with PyTorch using the MovieLens 100K dataset. The project uses Neural Collaborative Filtering with learned user and movie embeddings to predict movie ratings and generate personalized recommendations.

Instead of utilizing similarity metrics or matrix factorization techniques, a neural network is trained to learn underlying representations/embeddings for users and movies directly from rating data. The model predicts how a user would rate a movie and uses those predictions to recommend movies the user hasn't seen yet. The recommendation system scores unseen movies and recommends the highest-rated candidates.

## Dataset
MovieLens 100K Dataset<br>
Source: https://grouplens.org/datasets/movielens/

Dataset Statistics:
- 100,000 ratings
- 943 users
- 1,682 movies
- Ratings from 1 to 5 stars

## Technologies Used
- Python
- PyTorch
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

## Model Architecture
### Baseline Model: Neural Collaborative Filtering
<img width="484" height="1106" alt="Neural Network Movie Rating Prediction" src="https://github.com/user-attachments/assets/edc7a14b-5d6d-4203-a807-095120a4f23b" />

### Embeddings
The model learns:
- A 50-dimensional embedding for each user
- A 50-dimensional embedding for each movie
These embeddings capture latent user preferences and movie characteristics directly from the rating data.

## Training Process
### Loss Function
Mean Squared Error (MSE)

### Optimizer
Adam Optimizer

### Training Loop
For each batch:
1. Generate predictions
2. Compute loss
3. Perform backpropagation
4. Update model weights

## Results
### Baseline Collaborative Filtering Model
<b>Training Loss (5 Epochs):</b> <br>
Epoch 1: 1.6571 <br>
Epoch 2: 1.0009 <br>
Epoch 3: 0.9250 <br>
Epoch 4: 0.8827 <br>
Epoch 5: 0.8549 <br>

<b>Final Test Loss:</b> 0.9481

<b>Sample Predictions:</b> <br>
Actual: 1.0, Predicted: 2.82 <br>
Actual: 2.0, Predicted: 2.95 <br>
Actual: 3.0, Predicted: 3.57 <br>
Actual: 4.0, Predicted: 3.96 <br>
Actual: 4.0, Predicted: 3.93 <br>

## Generating Recommendations
After training, the model:
1. Identifies movies a user has not rated
2. Predicts ratings for those movies
3. Returns the highest-scoring candidates

## Embedding Visualization
Movie embeddings were projected from 50 dimensions to 2 dimensions using t-SNE. This visualization shows inspection of how the network organizes movies within the learned embedding space.

Insights:
- Some highly-rated classic films clustered together
- Similarity patterns emerged from user behavior rather than explicit genre labels
- Learned representations didn't always align with assumptions about movie similarity

## Hybrid Recommendation Model
As an experiment, movie genre information was incorporated into the model.<br>
Updated architecture:<br>
<img width="692" height="1104" alt="Neural Network Movie Rating Prediction (1)" src="https://github.com/user-attachments/assets/9be84b21-6148-4018-8c40-879238437503" />

## Experiment Results
### Hypothesis
Adding movie genre information would improve recommendation quality and test performance

### Results
| Model | Test Loss |
|--------|-----------|
| Baseline Collaborative Filtering | 0.9481 |
| Hybrid Recommender + Genres | 0.9431 |

### Conclusion
Adding genre information provided little to no improvement. 

The learned user and movie embeddings appeared to capture much of the useful information already present in the dataset.

## Overfitting Analysis
The hybrid model was trained for 20 epochs to evaluate the effect of longer training.

Results:<br>
Training Loss: 0.8463 -> 0.5505<br>
Test Loss: 0.9431 -> 1.0808<br>

Although training performance significantly improved, test performance degraded. This demonstrates overfitting, where the model memorizes training data instead of learning patterns that generalize to unseen examples.

### Key Takeaways
Lower training loss doesn't necessarily mean a better model. Evaluation on held-out test data is critical.
