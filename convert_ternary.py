def convert(fileName):
    imageFile = open(fileName,'r')
    lines = imageFile.readlines()
    imageFile.close()
    numbers = []
    for line in lines:
        line = line.strip('\n')
        eachNumber = []
        if line != None:
            for i in range(len(line)):
                if line[i] == ' ':
                    eachNumber.append('0')
                elif line[i] == '#':
                    eachNumber.append('2')
                else:
                    eachNumber.append('1')
            numbers.append(eachNumber)

    vectorFile = open('testIamageOutput2.txt','w')
    for number in numbers:
        for pixel in number:
            vectorFile.write(pixel)
        vectorFile.write('\n')
    vectorFile.close()

def main():
    convert("testimages")

if __name__ == "__main__":
    main()
