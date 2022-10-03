class CustomGenerator:
    def __init__(self, input_list, /, input_map_func=lambda it: it):
        self.__input_list = input_list
        self.__input_map_func = input_map_func

    def __len__(self):
        return len(self.__input_list)

    def __getitem__(self, ind):
        if isinstance(ind, int):
            return self.__input_map_func(self.__input_list[ind])
        else:
            return [self.__input_map_func(self.__input_list[index]) for index in ind]
