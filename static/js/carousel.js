document.addEventListener('DOMContentLoaded', function() {
    // Находим все контейнеры карусели
    const carouselContainers = document.querySelectorAll('.carousel-container');
    
    carouselContainers.forEach(container => {
        const scrollContainer = container.querySelector('.movie-scroll-container');
        const leftBtn = container.querySelector('.carousel-btn-left');
        const rightBtn = container.querySelector('.carousel-btn-right');
        const movieRow = container.querySelector('.movie-row');
        
        if (!scrollContainer || !leftBtn || !rightBtn || !movieRow) return;
        
        // Функция для обновления видимости стрелок
        function updateArrowVisibility() {
            const scrollLeft = scrollContainer.scrollLeft;
            const scrollWidth = scrollContainer.scrollWidth;
            const clientWidth = scrollContainer.clientWidth;
            
            // Показываем/скрываем левую стрелку
            if (scrollLeft <= 10) {
                leftBtn.style.opacity = '0';
                leftBtn.style.visibility = 'hidden';
                leftBtn.style.pointerEvents = 'none';
            } else {
                leftBtn.style.opacity = '1';
                leftBtn.style.visibility = 'visible';
                leftBtn.style.pointerEvents = 'auto';
            }
            
            // Показываем/скрываем правую стрелку
            if (scrollLeft + clientWidth >= scrollWidth - 10) {
                rightBtn.style.opacity = '0';
                rightBtn.style.visibility = 'hidden';
                rightBtn.style.pointerEvents = 'none';
            } else {
                rightBtn.style.opacity = '1';
                rightBtn.style.visibility = 'visible';
                rightBtn.style.pointerEvents = 'auto';
            }
        }
        
        // Прокрутка влево
        leftBtn.addEventListener('click', function() {
            const cardWidth = movieRow.querySelector('.movie-card')?.offsetWidth || 300;
            const gap = parseInt(getComputedStyle(movieRow).gap) || 24;
            const scrollAmount = cardWidth + gap;
            
            scrollContainer.scrollBy({
                left: -scrollAmount,
                behavior: 'smooth'
            });
        });
        
        // Прокрутка вправо
        rightBtn.addEventListener('click', function() {
            const cardWidth = movieRow.querySelector('.movie-card')?.offsetWidth || 300;
            const gap = parseInt(getComputedStyle(movieRow).gap) || 24;
            const scrollAmount = cardWidth + gap;
            
            scrollContainer.scrollBy({
                left: scrollAmount,
                behavior: 'smooth'
            });
        });
        
        // Обновляем видимость стрелок при прокрутке
        scrollContainer.addEventListener('scroll', updateArrowVisibility);
        
        // Обновляем видимость стрелок при изменении размера окна
        window.addEventListener('resize', updateArrowVisibility);
        
        // Инициализируем видимость стрелок
        setTimeout(updateArrowVisibility, 100);
        
        // Также обновляем видимость при наведении на контейнер
        container.addEventListener('mouseenter', function() {
            updateArrowVisibility();
        });
    });
});