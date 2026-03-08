import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_data(filename):
	x = []
	y = []
	
	with open(os.path.join(BASE_DIR, filename), 'r', newline='') as file:
		reader = csv.DictReader(file)
		
		for row in reader:
			x.append(float(row['RPS']))
			y.append(float(row['CPU']))
			
	return x, y

def read_data_from_txt(name_file):
	x = []
	y = []
	
	with open(os.path.join(BASE_DIR, name_file), 'r', encoding="utf-8") as file:
		raw = file.read().splitlines()
		for line in raw[3:]:  # Пропускаємо перші 3 рядки
			parts = line.split('|')
			if len(parts) == 3:
				# print(parts)
				x.append(float(parts[1].strip()))
				y.append(float(parts[2].strip()))
			
	return x, y

def main():
    x, y = read_data("data.csv")

    n = len(x)
    h = (x[n-1] - x[0])/n

    print(f"Кількість вузлів: {n}")
    print("\nТабуляція вузлів:")
    print(" № | RPS | CPU")
    with open(os.path.join(BASE_DIR, "data.txt"), "w", encoding="utf-8") as f:
        f.write(f"Кількість вузлів: {n}\n")
        f.write("Табуляція вузлів:\n")
        f.write(" № | RPS | CPU\n")
        for i, point in enumerate(x):
            line = (f"{i:2d} | {x[i]:.2f} | {y[i]:.2f}\n")
            f.write(line)
            print(line, end="")

    print("x:", x)
    print("y:", y)


if __name__ == "__main__":
    main()