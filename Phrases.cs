using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using Random = UnityEngine.Random;

public class Phrases : MonoBehaviour
{
    // Храним только английские фразы для генерации текста
    [System.NonSerialized] public List<string> phrases = new List<string>
    {
        "Hello world",
        "How are you",
        "I am fine",
        "Good morning",
        "Nice to meet you",
        "What is your name",
        "My name is AI",
        "How old are you",
        "I love programming",
        "The weather is nice",
        "Have a good day",
        "See you later",
        "Thank you very much",
        "You are welcome",
        "What time is it",
        "Where are you from",
        "I am from Russia",
        "Do you speak English",
        "Yes I do",
        "No I don't",
        "Please help me",
        "I need help",
        "Let's go together",
        "That sounds good",
        "I think so too",
        "Happy birthday to you",
        "Merry Christmas",
        "Happy new year",
        "Good night sweet dreams",
        "See you tomorrow"
    };
    
    public int length;
    public int lengthOfknown = 1;
    private int setLength = 1;
    public bool changeWord = false;
    public int getWord = 0;
    
    void Start()
    {
        length = phrases.Count;
        Debug.Log($"Phrases initialized: {length} English phrases, lengthOfknown: {lengthOfknown}");
    }
    
    void Update(){}
    
    public List<List<string>> GetAllPhrases()
    {
        List<List<string>> allPhrases = new List<List<string>>();
        
        // Для генератора текста используем только английские фразы
        for (int i = 0; i < phrases.Count; i++)
        {
            List<string> phraseItem = new List<string>
            {
                phrases[i], // Английская фраза как вход
                phrases[i]  // И та же фраза как цель для обучения
            };
            allPhrases.Add(phraseItem);
        }
        
        Debug.Log($"Loaded {allPhrases.Count} phrases for text generation");
        return allPhrases;
    }
    
    public List<string> GetPhrase()
    {
        int phrasesCount = phrases.Count;
        
        // Всегда ограничиваем lengthOfknown количеством фраз
        if (lengthOfknown > phrasesCount)
        {
            lengthOfknown = phrasesCount;
        }
        
        if (lengthOfknown > setLength || changeWord)
        {
            string ifNew = "No";
            
            if (changeWord)
            {
                changeWord = false;
                // Увеличиваем количество известных фраз
                if (lengthOfknown < phrasesCount)
                {
                    ++lengthOfknown;
                }
            }
            
            int choosePhraseIndex;
            if (Random.Range(0, 10) >= 7)
            {
                // Берем новую фразу (последнюю из известных)
                choosePhraseIndex = lengthOfknown - 1;
                ifNew = "Yes";
            }
            else
            {
                // Берем случайную фразу из уже известных
                choosePhraseIndex = Random.Range(0, lengthOfknown);
            }
            
            // Проверяем границы
            if (choosePhraseIndex >= phrasesCount) choosePhraseIndex = phrasesCount - 1;
            
            string chosenPhrase = phrases[choosePhraseIndex];
            
            return new List<string> { 
                chosenPhrase, // Входная фраза
                chosenPhrase, // Целевая фраза (та же самая для обучения автогенерации)
                ifNew 
            };
        }
        else
        {
            // Берем случайную фразу из известных
            int choosePhraseIndex = Random.Range(0, lengthOfknown);
            
            // Проверяем границы
            if (choosePhraseIndex < 0) choosePhraseIndex = 0;
            if (choosePhraseIndex >= phrasesCount) choosePhraseIndex = phrasesCount - 1;
            
            string chosenPhrase = phrases[choosePhraseIndex];
            
            return new List<string> { 
                chosenPhrase, 
                chosenPhrase,
                "Yes"
            };
        }
    }
    
    // Вспомогательный метод для отладки
    public string GetDebugInfo()
    {
        return $"Total phrases: {phrases.Count}, lengthOfknown: {lengthOfknown}";
    }
}