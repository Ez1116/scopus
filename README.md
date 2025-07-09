# README.md

## 項目概述

這是一個Python腳本，用於分析學術文章摘要是否與研究者關心的主題有關。它使用Google Gemini的API對摘要進行0-5分的相關性評分，可藉由config.json進行主題與評分的調整

## 依賴和設置

安裝所需的軟件包：
```bash
pip install --upgrade google-genai pandas python-dotenv tqdm
```

環境設置：
- 設置 `GEMINI_API_KEY` 環境變量 或者 創建包含 `GEMINI_API_KEY=your-gemini-api-key` 的 `.env` 文件
- 腳本使用Google的 "gemini-2.0-flash" 模型

## 使用方法

運行腳本：
```bash
python scopus-abstract.py --input <csv_file> --output <output_file> --config <config_file>
```

**使用預設GenAI評量配置：**
```bash
python scopus-abstract.py --input scopus_AI.csv --output scopus_AI_evaluations.csv
```

**使用自訂配置：**
```bash
python scopus-abstract.py --input my_data.csv --output my_results.csv --config my_custom_config.json
```

**輸入CSV要求：**
- 必須包含 'title' & 'abstract' & 'DOI' & 'Link' 列

**輸出CSV格式：**
- `title`：文章標題
- `relevance_score`：0-5的整數相關性評分
- `DOI`：文章永久網址
- `Link`：文章 SCOPUS 連結

## 配置系統

腳本現在支持通過JSON文件進行靈活配置：

**預設配置：** `config_genai_assessment.json` - GenAI評量研究的預配置
**模板配置：** `config_template.json` - 創建自定義研究配置的模板

**配置文件結構：**
- `research_topic`：研究焦點描述
- `tool_name`：評估函數名稱（通常為"record_evaluation"）
- `tool_description`：評估工具功能說明
- `scoring_criteria`：詳細的0-5評分標準
- `system_prompt_template`：AI模型指示
- `evaluation_instruction`：評估用戶指示

## 架構

**單文件設計**，主要組件包括：
- **批次處理**：在單個API請求中處理所有文章以提高效率
- **Gemini集成**：使用JSON響應解析獲得結構化評分
- **配置加載**：基於JSON的靈活主題配置
- **錯誤處理**：優雅地處理API響應解析失敗
- **進度跟踪**：為批次處理提供簡單的進度反饋

**關鍵常量：**
- `MODEL_NAME = "gemini-2.0-flash"`
- `CONCURRENT_REQUESTS = 50`（遺留，不再使用）
- `MAX_TOKENS = 8192`

## 自訂不同研究主題

要將腳本適應不同的研究主題：

1. **複製模板：** `cp config_template.json my_research_config.json`
2. **編輯配置：** 修改所有字段以匹配您的研究焦點
3. **使用自訂配置運行：** `python scopus-abstract.py --input data.csv --output results.csv --config my_research_config.json`

## 開發注意事項

- 沒有正式的測試套件
- 依賴項僅在代碼註釋中記錄
- 配置現在外部化在JSON文件中
- 錯誤處理在API失敗時默認評分為0
- 使用Google Gemini的函數調用功能獲得結構化響應
- 配置驗證確保所有必需字段都存在
