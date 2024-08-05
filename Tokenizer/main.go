package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {
	action := os.Args[1]
	if action == "train" {
		filename, comp := os.Args[2], os.Args[3]
		compression, err := strconv.Atoi(comp)
		if err != nil {
			fmt.Println(err)
			return
		}
		train(filename, compression)
	} else {
		tokens := tokenize(os.Args[2])
		for i := range tokens {
			fmt.Print(tokens[i], " - ")
		}
	}

}
