all:
	gcc FFNN.c main.c -o out.exe

clean:
	-rm -f out.exe

