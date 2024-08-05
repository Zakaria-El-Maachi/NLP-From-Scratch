package main

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"
)

type Rule struct {
	A string
	B string
}

// func wordpiece(compression int, tokens []string) []Rule

func bpeTrainingIteration(frequencies map[string]int, tokens []string) int {
	best := -1
	maxCount := 0
	for i := 0; i < len(tokens)-1; i++ {
		if tokens[i] == " " || tokens[i+1] == " " || isPunctuation(tokens[i]) || isPunctuation(tokens[i+1]) {
			continue
		}
		temp := tokens[i] + tokens[i+1]
		frequencies[temp] += 1
		if frequencies[temp] > maxCount {
			best = i
			maxCount = frequencies[temp]
		}
	}
	return best
}

func bpe(compression int, tokens []string) ([]string, []Rule) {

	var frequencies map[string]int
	var best int
	rules := []Rule{}
	var newRule Rule

	for k := 0; k < compression; k++ {
		frequencies = make(map[string]int)
		best = bpeTrainingIteration(frequencies, tokens)
		if best == -1 {
			break
		}
		// fmt.Println(best)
		newRule = Rule{tokens[best], tokens[best+1]}
		rules = append(rules, newRule)
		tokens = merge(tokens, newRule)
	}

	return tokens, rules
}

func train(filename string, compression int) {
	// Read File
	tokens := readFile(filename)

	// Apply BPE Algorithm
	tokens, rules := bpe(compression, tokens)
	fmt.Println("Number of Tokens : ", len(tokens), "\nNumber of Rules : ", len(rules))

	// Save the Rules
	ruleFile, err := os.Create("rules.gob")
	if err != nil {
		log.Fatalln(err)
	}
	encoder := gob.NewEncoder(ruleFile)

	if err = encoder.Encode(rules); err != nil {
		log.Fatalln(err)
	}
}
