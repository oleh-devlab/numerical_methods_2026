import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_data(filename):
	x = []
	y = []
	
	with open(os.path.join(BASE_DIR, filename), 'r', newline='') as file:
		reader = csv.DictReader(file)
		
		for row in reader:
			x.append(float(row['Month']))
			y.append(float(row['Temp']))
			
	return x, y

def main():
    x, y = read_data("data.csv")

    print("x:", x)
    print("y:", y)


if __name__ == "__main__":
    main()