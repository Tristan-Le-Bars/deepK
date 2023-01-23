SRC = SequentialModel.cpp \
	  LossFunctions.cpp \
	  DataLoader.cpp \
	  ActivationFunctions.cpp

LIB_NAME = deepK.a

OBJ = $(SRC:.cpp=.o)

deepK_test:
	g++ -o deepK_test Main.cpp $(SRC) -DWITHOUT_NUMPY -I/usr/local/lib/python3.10 -lpython3.10

deepK_lib:
	gcc -c $(SRC) -DWITHOUT_NUMPY -I/usr/local/lib/python3.10 -lpython3.10
	ar rcs $(LIB_NAME) $(OBJ)

.PHONY: clean

clean:
	rm deepK_test