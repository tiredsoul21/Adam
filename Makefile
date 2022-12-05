CC = g++ -g -Wall -std=c++17
SHELL := /bin/bash

# Project List
PROJECT_ADAM = Adam

ML = ./src/ml
PROJECT_BUILD = build/project

ADAM_HEADERS = $(ML)/types/inc/neuralTypes.hpp \
               $(ML)/neural_network/inc/neuron.hpp \
			   $(ML)/neural_network/inc/neuralLayer.hpp \
			   $(ML)/neural_network/inc/neuralNetwork.hpp \
               $(ML)/neural_network/inc/neuralNetworkTrainer.hpp \


all: buildAdam 

buildAdam: $(eval TARGET=$(PROJECT_ADAM)) \
		   $(eval BUILD_HEADERS=$(ADAM_HEADERS)) \
		   $(eval BUILD_DIR=$(PROJECT_BUILD)/$(TARGET)) \
		   buildObjects \
		   buildApplication

buildObjects: $(BUILD_HEADERS)
	@echo
	@echo "-----------------------------------"
	@echo "Beginning Object Build $(TARGET)"
	@echo "-----------------------------------"
	@echo $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)
	@for inc in $^ ; do \
	    src=`echo $${inc} | sed 's:/inc/:/src/:g' | sed 's:\.hpp:.cpp:g'`; \
	    obj=`echo $${inc} | sed -e 's:^.*/inc/:$(BUILD_DIR)/:g' | sed 's:\.hpp:.o:g'`; \
	    cp $${inc} $(BUILD_DIR); \
	    if [ -f "$${src}" ]; then \
	        echo "Building target: $${obj}"; \
	        $(CC) -c $${src} -o $${obj} -I $(BUILD_DIR); \
	    fi \
	done 

buildApplication:
	@echo
	@echo "-----------------------------------"
	@echo "Beginning Application Build $(TARGET)"
	@echo "-----------------------------------"
	$(CC) -I $(BUILD_DIR) src/main.cpp -o $(TARGET) $(BUILD_DIR)/*.o
	@mkdir -p dist
	@mv $(TARGET) dist

clean:
	@rm -rf ${BUILD_DIR}
	@rm -rf dist