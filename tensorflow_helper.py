def plot_loss_curves(history):
  """
  Plots the loss curve and accuracy for a given model

  Args:
    history: the history object returned by the fit method of a Keras model
  """

  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]
  epochs = range(len(loss))

  # plot loss
  plt.plot(epochs, loss, label="Training loss")
  plt.plot(epochs, val_loss, label="Validation loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  #plt.ylabel("")
  plt.legend()
  
  # plot accuracy
  plt.plot(epochs, accuracy, label="Training accuracy")
  plt.plot(epochs, val_accuracy, label="Validation accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  #plt.ylabel("")
  plt.legend()
  
  plt.show()
