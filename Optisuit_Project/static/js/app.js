document.addEventListener("DOMContentLoaded", () => {
    const activeNav = document.querySelector(".nav-links a.active");
    if (activeNav) {
        activeNav.setAttribute("aria-current", "page");
    }

    const tables = document.querySelectorAll("table");
    tables.forEach((table) => {
        table.setAttribute("role", "table");
    });

    const explainButtons = document.querySelectorAll(".explain-toggle");
    explainButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const targetId = button.getAttribute("data-target");
            const panel = document.getElementById(targetId);
            if (!panel) {
                return;
            }

            const isOpen = panel.classList.toggle("open");
            button.setAttribute("aria-expanded", isOpen ? "true" : "false");
            button.textContent = isOpen ? "Hide Explanation" : "What does this mean?";
        });
    });
});
