const header = document.querySelector("[data-header]");
const navToggle = document.querySelector(".nav-toggle");
const nav = document.querySelector(".site-nav");

function updateHeader() {
  header?.classList.toggle("scrolled", window.scrollY > 12);
}

updateHeader();
window.addEventListener("scroll", updateHeader, { passive: true });

navToggle?.addEventListener("click", () => {
  const isOpen = nav.classList.toggle("open");
  navToggle.setAttribute("aria-expanded", String(isOpen));
});

nav?.querySelectorAll("a").forEach((link) => {
  link.addEventListener("click", () => {
    nav.classList.remove("open");
    navToggle?.setAttribute("aria-expanded", "false");
  });
});

document.querySelectorAll("[data-tabs]").forEach((tabs) => {
  const tabButtons = [...tabs.querySelectorAll('[role="tab"]')];
  const panels = [...tabs.querySelectorAll('[role="tabpanel"]')];

  function activateTab(selected) {
    tabButtons.forEach((button) => {
      const active = button === selected;
      button.setAttribute("aria-selected", String(active));
      button.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.id !== selected.getAttribute("aria-controls");
    });
  }

  tabButtons.forEach((button, index) => {
    button.addEventListener("click", () => activateTab(button));
    button.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      let next = index;
      if (event.key === "ArrowRight") next = (index + 1) % tabButtons.length;
      if (event.key === "ArrowLeft") next = (index - 1 + tabButtons.length) % tabButtons.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabButtons.length - 1;
      activateTab(tabButtons[next]);
      tabButtons[next].focus();
    });
  });
});

async function copyText(value, button) {
  try {
    await navigator.clipboard.writeText(value);
    const previous = button.textContent;
    if (button.classList.contains("copy-button")) {
      button.setAttribute("aria-label", "Copied");
      button.style.color = "#1f6b4f";
    } else {
      button.textContent = "Copied";
    }
    window.setTimeout(() => {
      if (button.classList.contains("copy-button")) {
        button.setAttribute("aria-label", "Copy install command");
        button.style.color = "";
      } else {
        button.textContent = previous;
      }
    }, 1400);
  } catch {
    button.textContent = "Select & copy";
  }
}

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", () => copyText(button.dataset.copy, button));
});

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (target) copyText(target.innerText, button);
  });
});
