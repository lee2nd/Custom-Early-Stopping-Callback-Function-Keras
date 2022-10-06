from keras.callbacks import Callback

class EarlyStoppingByLimits(Callback):
	def __init__(self, train_limit=0.5, val_limit=0.5):
		super(Callback, self).__init__()
		self.train_limit = train_limit
		self.val_limit = val_limit

	def on_epoch_end(self, epoch, logs={}):
		loss = logs.get('loss')
		val_loss = logs.get('val_loss')
		if loss < self.train_limit or val_loss < self.val_limit:
			self.model.stop_training = True
      
## used like this
early_stop = EarlyStoppingByLimits (train_limit=0.75, val_limit=0.85)
model.fit(train_ds, epochs=100, validation_data=valid_ds, callbacks=[early_stop])
