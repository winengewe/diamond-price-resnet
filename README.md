# 💎 Diamond Price Prediction using Deep ResNet-MLP

A robust Deep Learning pipeline designed to predict the price of diamonds based on their physical attributes (Carat, Cut, Color, Clarity, and dimensions). 

Unlike standard regression tutorials, this project implements **professional MLOps best practices**, including target engineering (Log-Norm), a custom Residual Neural Network architecture, and a production-ready inference pipeline.

## 🚀 Key Features

* **Deep ResNet Architecture:** Implements a custom Keras model using **Residual Connections (Skip Connections)** and **Layer Normalization** to prevent vanishing gradients and allow for deeper network training.
* **Target Engineering:** Applies **Logarithmic Transformation (`np.log1p`)** to the target variable (Price) to handle the right-skewed financial distribution, significantly improving convergence.
* **Advanced Preprocessing:** Uses Scikit-Learn's `ColumnTransformer` pipeline to handle:
    * *Categorical Data:* One-Hot Encoding (Cut, Color, Clarity).
    * *Numerical Data:* Standard Scaling (Carat, Depth, Table, x, y, z).
* **Training Dynamics:**
    * **Cosine Decay Learning Rate:** Smoothly anneals the learning rate for better local minima convergence.
    * **Callbacks:** Implements `EarlyStopping` and `ModelCheckpoint` to prevent overfitting and save the best weights.
* **Inference Pipeline:** Includes a standalone function to process raw dictionary inputs, transforming them on-the-fly for real-world predictions.

## 🛠️ Tech Stack

* **Core:** Python 3.x
* **Deep Learning:** TensorFlow / Keras (Functional API)
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

## 🧠 Model Architecture

The model utilizes a **ResNet-MLP (Multi-Layer Perceptron)** design:
1.  **Input Layer:** Processes the transformed feature vector (approx. 26 dimensions).
2.  **Projection Layer:** Expands dimensions to the base neuron size.
3.  **Residual Blocks (x2):** * Layer Normalization -> Dense (Expand) -> GELU Activation -> Dense (Contract) -> Dropout.
    * **Skip Connection:** Adds the input of the block to the output (`layers.Add()`).
4.  **Output Layer:** Linear activation for regression (converting Log-Space predictions back to Dollar-Space).

## 📊 Performance

The model evaluates performance using **MAE (Mean Absolute Error)** and **R² Score**.

* **R² Score:** ~0.98+ (Indicates highly accurate trend capturing)
* **Test MAE:** Calculated in real dollars (post-inverse-log transformation).

*(Note: Exact metrics depend on the random seed and training duration).*

## 📂 Project Structure

* `main.py`: The complete training, evaluation, and inference script.
* `preprocessor.pkl`: Saved Scikit-Learn pipeline (generated after running).
* `best_diamond_model.keras`: The best performing model weights (generated after running).

## 💻 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/diamond-price-resnet.git](https://github.com/YOUR_USERNAME/diamond-price-resnet.git)
    cd diamond-price-resnet
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
    ```

3.  **Run the script:**
    ```bash
    python main.py
    ```

## 🔮 Usage Example

The code includes a demo inference function. To predict the price of a new diamond:

```python
sample_diamond = {
    'carat': 1.21,
    'cut': 'Very Good',
    'color': 'E',
    'clarity': 'SI1',
    'depth': 62.2,
    'table': 60,
    'x': 6.78, 'y': 6.76, 'z': 4.21
}

# The script automatically handles loading artifacts and predicting
# Result: Predicted Price: $7,703.xx
