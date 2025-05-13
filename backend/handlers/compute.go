package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

type RequestData struct {
	Function string `json:"function"`
}

func ComputeHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Метод не поддерживается", http.StatusMethodNotAllowed)
		return
	}

	var requestData RequestData
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Ошибка чтения тела запроса", http.StatusBadRequest)
		return
	}
	fmt.Println("[LOG] Тело запроса:", string(body))

	if err := json.Unmarshal(body, &requestData); err != nil {
		http.Error(w, "Ошибка разбора JSON", http.StatusBadRequest)
		return
	}
	fmt.Println("[LOG] Получена передаточная функция:", requestData.Function)

	workingDir, err := os.Getwd()
	if err != nil {
		http.Error(w, "Ошибка получения текущей директории", http.StatusInternalServerError)
		return
	}

	scriptPath := filepath.Join(workingDir, "compute", "main.py")
	staticDir := filepath.Join(workingDir, "static")

	fmt.Println("[LOG] Путь к Python-скрипту:", scriptPath)
	fmt.Println("[LOG] Путь к директории static:", staticDir)

	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		http.Error(w, "main.py не найден", http.StatusInternalServerError)
		return
	}

	if _, err := os.Stat(staticDir); os.IsNotExist(err) {
		if err := os.MkdirAll(staticDir, 0755); err != nil {
			http.Error(w, "Ошибка создания директории static", http.StatusInternalServerError)
			return
		}
	}

	cmd := exec.Command("python3", scriptPath, requestData.Function, staticDir)

	// Разделяем stdout и stderr
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err = cmd.Run()

	// Логируем stderr (логи Python)
	fmt.Println("[PYTHON LOGS]:", stderr.String())

	if err != nil {
		http.Error(w, "Ошибка выполнения Python-скрипта:\n"+stderr.String(), http.StatusInternalServerError)
		return
	}

	// Парсим только stdout (где должен быть чистый JSON)
	outputStr := strings.TrimSpace(stdout.String())
	var response map[string]interface{}
	if err := json.Unmarshal([]byte(outputStr), &response); err != nil {
		http.Error(w,
			"Ошибка парсинга JSON от Python:\n"+
				"Ошибка: "+err.Error()+"\n"+
				"Вывод Python (stdout):\n"+outputStr+"\n"+
				"Логи Python (stderr):\n"+stderr.String(),
			http.StatusInternalServerError,
		)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
