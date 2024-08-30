import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from plotly import graph_objects as go

df = pd.read_csv('data/dataset.csv')


train_df = df.sample(frac=0.75, random_state=42)
test_df = df.drop(train_df.index)
N_train = len(train_df)
N_test = len(test_df)


#Normalization
max_val = train_df.max(axis=0)
min_val = train_df.min(axis=0)
range = max_val - min_val
train_df = (train_df - min_val)/range
test_df = (test_df - min_val)/range


X_train = train_df.drop('label',axis=1)
y_train = train_df['label']
X_test = test_df.drop('label',axis=1)
y_test = test_df['label']
ncol = [X_train.shape[1]]


neural_net = tf.keras.Sequential([
    tf.keras.layers.Dense(units=32, activation='relu',input_shape=ncol),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='relu'),
    # tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=1)
])
print(neural_net.summary())


neural_net.compile(optimizer='adam',loss='mse', metrics=['accuracy'])

losses = neural_net.fit(X_train, y_train,
				validation_data=(X_test, y_test),
				batch_size=N_train//10, 
				epochs=100, # total epoch
)

loss, accuracy = neural_net.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")


models = [neural_net]
validation_losses = [losses.history['val_loss'][-1]]

# Encontra o índice do melhor modelo
best_model_index = validation_losses.index(min(validation_losses))

# Salva o melhor modelo
best_model = models[best_model_index]
best_model.save_weights('best_model_weights.h5')

loss_df = pd.DataFrame(losses.history)
loss_df.loc[:,['loss','val_loss']].plot()


X = X_test.iloc[:,:]
x1 = (X.iloc[:,0]).to_numpy()
x2 = (X.iloc[:,1]).to_numpy()
y = (y_test.iloc[:]).to_numpy()
ypred = neural_net.predict(X)
ypred = ypred.reshape((len(ypred),))

fig = go.Figure(data=[go.Scatter3d(
    x=x1,
    y=x2,
    z=y,
    mode='markers',
    marker=dict(
        size=8,
        color='blue',
        symbol='circle'
    )
)])

fig.add_trace(go.Scatter3d(
    x=x1,
    y=x2,
    z=ypred,
    mode='markers',
    marker=dict(
        size=8,
        color='red',
        symbol='circle'
    )
))

fig.update_layout(scene=dict(
    xaxis_title='Eixo X',
    yaxis_title='Eixo Y',
    zaxis_title='Eixo Z',
    camera_eye=dict(x=1.25, y=1.25, z=1.25)
))

fig.show()