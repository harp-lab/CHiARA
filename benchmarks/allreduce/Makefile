# --------------------------------------------------
# Compiler and flags
# --------------------------------------------------
CXX       := mpicxx
CXXFLAGS  := -O3 -std=c++17 -Wall 
LDFLAGS   :=

# --------------------------------------------------
# Target
# --------------------------------------------------
TARGET := out

# --------------------------------------------------
# Sources
# --------------------------------------------------
SRCS := \
    main.cpp \
    allreduce_ring.cpp \
    allreduce_reduce_scatter_allgather.cpp \
    allreduce_recursive_doubling.cpp \
    allreduce_recursive_multiplying.cpp \
    allreduce_recexch.cpp \
    allreduce_k_reduce_scatter_allgather.cpp 

OBJS := $(SRCS:.cpp=.o)

# --------------------------------------------------
# Rules
# --------------------------------------------------
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp common.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean