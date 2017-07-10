from keras.models import Sequential

model = Sequential()

# Add input layer and first hidden layer
model.add(Dense(units=64, input_dim=100))
model.add(Activation('relu'))

# Add output layer
model.add(Dense(units=10))
model.add(Activation('softmax'))

# Configure optimizer
model.compile(loss=keras.losses.categorical_crossentropy, ...
	optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
	
# Train network with set data
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Train for one batch
#model.train_on_batch(x_batch, y_batch)

# Evaluate performance
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

# Predict on new data
classes = model.predict(x_test, batch_size=128)