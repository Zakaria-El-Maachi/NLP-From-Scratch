package main

import (
	"encoding/gob"
	"log"
	"os"
)

func tokenize(filename string) []string {
	// Read File
	tokens := readFile(filename)

	// Read Rules
	ruleFile, err := os.Open("rules.gob")
	if err != nil {
		log.Fatalln(err)
	}
	decoder := gob.NewDecoder(ruleFile)

	var rules []Rule
	if err = decoder.Decode(&rules); err != nil {
		log.Fatalln(err)
	}

	for i := range rules {
		tokens = merge(tokens, rules[i])
	}

	return tokens
}
