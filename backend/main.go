package main

import (
	"fmt"
	"log"
	"net/http"
	"path/filepath"
	"project/backend/handlers"

	"github.com/rs/cors"
)

// обёртка для установки заголовков "no-cache"
func noCacheWrapper(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
		w.Header().Set("Expires", "0")
		h.ServeHTTP(w, r)
	})
}

func main() {
	// Раздача статических файлов из папки "static" для изображений
	fsStatic := http.FileServer(http.Dir("./static"))
	http.Handle("/static/", noCacheWrapper(http.StripPrefix("/static/", fsStatic)))

	// Раздача фронтенда (HTML, CSS, JS) из папки "frontend"
	http.Handle("/frontend/", http.StripPrefix("/frontend/", http.FileServer(http.Dir("./frontend"))))

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		indexPath := filepath.Join("frontend", "index.html")
		http.ServeFile(w, r, indexPath)
	})

	// Обработчик API для вычислений
	http.HandleFunc("/api/compute", handlers.ComputeHandler)

	// CORS middleware
	corsHandler := cors.New(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "OPTIONS"},
		AllowedHeaders:   []string{"Content-Type", "Authorization"},
		AllowCredentials: true,
	})

	// Оборачиваем маршрутизатор в CORS
	handler := corsHandler.Handler(http.DefaultServeMux)

	// Запуск сервера
	fmt.Println("Server started on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", handler))
}
