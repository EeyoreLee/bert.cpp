#include "bert.h"

#include <string>
#include <iostream>
#include <vector>
#include <chrono>

int main(int argc, char **argv)
{
    const std::string path = "/media/E/lichunyu/bert.cpp/data/test_tokenizer/tokenizer.json";
    const std::string text1 = "测试一下tokenizer返回的结果是否和python一致";
    const std::string text2 = "测试一下tokenizer返回的结果是否和python";
    const std::string text3 = "测试一下tokenizer返回的结果是";
    const std::string text4 = "测试一下tokenizer";

    std::vector<std::string> batch_text{text1, text2, text3, text4};

    auto start_cpp = std::chrono::high_resolution_clock::now();

    bert_tokenizer *tokenizer = new bert_tokenizer();
    tokenizer->from_file(path);
    std::vector<std::vector<int>> bacth_ids = tokenizer->batch_encode(batch_text);
    for (auto ids : bacth_ids)
    {
        for (auto i : ids)
        {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    auto end_cpp = std::chrono::high_resolution_clock::now();
    auto duration_cpp = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpp - start_cpp).count();

    std::cout << "C++&Rust  cost: " << duration_cpp << "ms." << std::endl;

    // free
    delete tokenizer;

    return 0;
};