class ParseFile:
    def __init__(self, file_path):        
        self.t_speed = None
        self.d_speed = None
        self.no_nodes = None
        self.filepath = file_path
        self.X = []
        self.Y = []
        self.loc = []
        self.parse_instance()

    def check_numbers(self, el):
        try:
            float(el)
            return True
        except ValueError:
            return False       

    def read_file(self, file_path):
        data = []
        file = open(file_path,"r")
        lines = file.readlines()
        for l in lines:
            l = l.replace("\n","")
            l = l.split(" ")
            if self.check_numbers(l[0]):
                data.append(l)
        return data

    def parse_instance(self):
        data = self.read_file(self.filepath)
        for el in range(len(data)):
            if el == 0:
                self.t_speed = float(data[el][0])
            elif el == 1:
                self.d_speed = float(data[el][0])
            elif el == 2:
                self.no_nodes = int(data[el][0])
            else:
                self.X.append(float(data[el][0]))
                self.Y.append(float(data[el][1]))
                self.loc.append(data[el][2])

