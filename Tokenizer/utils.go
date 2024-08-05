package main

import (
	"log"
	"os"
	"strings"
	"unicode"
)

// func pretokenize() {

// }

func readFile(filename string) []string {
	fileContent, err := os.ReadFile(filename)
	if err != nil {
		log.Fatalln(err)
	}
	return strings.Split(string(fileContent), "")
}

func isPunctuation(char string) bool {
	if len(char) != 1 {
		return false
	}

	r := []rune(char)[0]
	return unicode.IsPunct(r)
}

func merge(tokens []string, rule Rule) []string {
	mergedWord := rule.A + rule.B
	for i := 0; i < len(tokens)-1; i++ {
		if tokens[i] == rule.A && tokens[i+1] == rule.B {
			tokens = append(tokens[:i], tokens[i+1:]...)
			tokens[i] = mergedWord
		}
	}
	return tokens
}
