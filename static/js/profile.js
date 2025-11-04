// Переключение между разделами профиля
        document.addEventListener('DOMContentLoaded', function() {
            const navBtns = document.querySelectorAll('.nav-btn');
            const contentSections = document.querySelectorAll('.content-section');
            
            navBtns.forEach(btn => {
                btn.addEventListener('click', function() {
                    // Убираем активный класс у всех кнопок и секций
                    navBtns.forEach(b => b.classList.remove('active'));
                    contentSections.forEach(s => s.classList.remove('active'));
                    
                    // Добавляем активный класс текущей кнопке
                    this.classList.add('active');
                    
                    // Показываем соответствующую секцию
                    const tabId = this.getAttribute('data-tab');
                    document.getElementById(tabId).classList.add('active');
                });
            });
        });