import tflearn.callbacks


class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, accuracy_threshold, loss_threshold, log_filename):
        super().__init__()
        self.__accuracy_threshold = accuracy_threshold
        self.__loss_threshold = loss_threshold
        self.__log_filename = log_filename
        # Clearing old logs, adding column names
        with open(self.__log_filename, 'w') as f:
            f.write('val_loss\tval_acc')

    def on_epoch_end(self, training_state):
        val_acc = training_state.val_acc
        val_loss = training_state.val_loss
        if val_acc is None or val_loss is None:
            return
        with open(self.__log_filename, 'a') as f:
            f.write(f'\n{val_loss}\t\t{val_acc}')
        if val_acc >= self.__accuracy_threshold and val_loss <= self.__loss_threshold:
            print('############################# EARLY STOP #############################')
            raise StopIteration()
        else:
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NO EARLY STOP... #############################')
