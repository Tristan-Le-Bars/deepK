SRC = test_lib.cpp

LIB = ./src/deepK.a

OBJ = $(SRC:.cpp=.o)

deepK_test:
	g++ -o test_lib $(SRC) $(LIB) -DWITHOUT_NUMPY -I/usr/local/lib/python3.10 -lpython3.10


clean:
	rm deepK_test