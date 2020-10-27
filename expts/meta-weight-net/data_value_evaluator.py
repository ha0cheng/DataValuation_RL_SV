import sys
import matplotlib


def drawPic(performances):
    pass

class DataValueEvaluator():
    '''

    '''

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        '''
        load model from local file
        '''

    def evaluate_most(self):
        '''

        '''
        most_performances = []
        # self.model ==> tarin_data
        # self.model ==> w_list
        # tarin_data.sort(key = w_list) 
        for up_boundary_rate in range(0, 0.5, 0.05):
            up_boundary = int(up_boundary_rate * len(train_data))
            # new_train_data = train_data[:boundary]
            # performance = model.train(data)
            # most_performances.append(performance)
        # draw pic
        pass


    def evaluate_least(self):
        '''
        '''
        least_performances = []
        # self.model ==> tarin_data
        # self.model ==> w_list
        # tarin_data.sort(key = w_list) 
        for low_boundary_rate in range(0, 0.5, 0.05):
            low_boundary = int(low_boundary_rate * len(train_data))
            # new_train_data = train_data[:boundary]
            # performance = model.train(data)
            # most_performances.append(performance)
        # draw pic
        pass





def main():
    data_value_evaluator = DataValueEvaluator("/..")

if __name__ == "__main__":
    sys.exit(main())