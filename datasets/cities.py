from csv import reader


class cities_loader():
    def __init__(self):
        with open(f'assets/cities.csv') as cities_file:
            for row in reader(cities_file):
                if len(row) != 3:
                    raise IndexError(f"Input row for row:'{row[0]}' len(row) != 3")
                city = row[0]
                location = (float(row[1]), float(row[2]))
                self.__dict__[city] = location
    
     
    def get_cities_list(self):
        lst = list(self.__dict__.keys())
        return lst
    
    def __getitem__(self, city):
        return self.__dict__[city]
    
    def __iter__(self):
        return iter(self.__dict__.items())