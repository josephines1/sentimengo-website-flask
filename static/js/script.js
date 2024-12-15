function tentangAnimation() {
  return {
    init() {
      this.setupIntersectionObserver();
    },
    setupIntersectionObserver() {
      const options = {
        root: null,
        rootMargin: "0px",
        threshold: 0.5,
      };

      const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            this.animateElements();
            observer.disconnect();
          }
        });
      }, options);

      observer.observe(this.$el);
    },
    animateElements() {
      // Animasi untuk Small Title (fadeInDown)
      this.$refs.tentangSmallTitleText.classList.add("animate-fadeInDown");

      // Animasi untuk Paragraf (fadeInUp)
      this.$refs.tentangParagrafText.classList.add("animate-fadeInUp");

      // Animasi typewriter untuk Solusi Presensi Modern
      const typewriterText = this.$refs.tentangTitleText;
      const text = typewriterText.innerText;
      typewriterText.innerText = "";
      let i = 0;
      const speed = 50; // Kecepatan ketikan dalam ms

      function typeWriter() {
        if (i < text.length) {
          typewriterText.classList.remove("opacity-0");
          typewriterText.innerHTML += text.charAt(i);
          i++;
          setTimeout(typeWriter, speed);
        }
      }
      typeWriter();
    },
  };
}

function tahapanDataAndAnimation() {
  return {
    init() {
      this.setupIntersectionObserver();
    },
    tahapanPoints: [
      {
        title: "Upload file CSV",
        description:
          "Upload file csv dengan kolom content, score, thumbsUpCount, reviewCreatedVersion, at, dan kota.",
      },
      {
        title: "Tinjau Data",
        description:
          "Tinjau data yang Anda unggah sebelum memulai proses analisis sentimen.",
      },
      {
        title: "Mulai Analisis",
        description:
          "Klik button Start Analyze untuk memulai proses analisis sentimen",
      },
      {
        title: "Unduh & Lihat Hasil Analisis",
        description:
          "Anda dapat mengunduh file csv hasil analisis sentimen dan juga melihat analisis data Anda di Power BI.",
      },
    ],
    setupIntersectionObserver() {
      const options = {
        root: null,
        rootMargin: "0px",
        threshold: 0.5,
      };

      const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            this.animateElements();
            observer.disconnect();
          }
        });
      }, options);

      observer.observe(this.$el);
    },
    animateElements() {
      const tahapanPointText = this.$refs.tahapanPointText;

      // Animasi untuk Small Title dan Title (fadeInDown)
      this.$refs.tahapanSmallTitleText.classList.add("animate-fadeInDown");
      this.$refs.tahapanTitleText.classList.add("animate-fadeInDown");

      // Animasi untuk tahapanPointText (fadeInUp)
      Array.from(tahapanPointText.children).forEach((element, index) => {
        setTimeout(() => {
          element.classList.add("animate-fadeInUp");
        }, index * 300);
      });
    },
  };
}

function teamDataAndAnimation() {
  return {
    init() {
      this.setupIntersectionObserver();
    },
    teamPoints: [
      {
        title: "Josephine",
        description: "Project Manager",
        imageURL:
          "https://acmindonesia.sharepoint.com/_layouts/15/userphoto.aspx?size=S&accountname=josephinesantoso9b%40member.maribelajar.org",
      },
      {
        title: "Fatmah",
        description: "Data Analyst",
        imageURL:
          "https://acmindonesia.sharepoint.com/_layouts/15/userphoto.aspx?size=S&accountname=fatmahsp08%40member.maribelajar.org",
      },
      {
        title: "Roro Pradnya Palupi",
        description: "UI/UX Designer",
        imageURL:
          "https://acmindonesia.sharepoint.com/_layouts/15/userphoto.aspx?size=S&accountname=anyaapalupi%40member.maribelajar.org",
      },
      {
        title: "Sekar Ayu Danastri",
        description: "Data Engineer",
        imageURL:
          "https://acmindonesia.sharepoint.com/_layouts/15/userphoto.aspx?size=S&accountname=danastri350%40member.maribelajar.org",
      },
    ],
    setupIntersectionObserver() {
      const options = {
        root: null,
        rootMargin: "0px",
        threshold: 0.5,
      };

      const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            this.animateElements();
            observer.disconnect();
          }
        });
      }, options);

      observer.observe(this.$el);
    },
    animateElements() {
      const teamPointText = this.$refs.teamPointText;

      // Animasi untuk Small Title dan Title (fadeInDown)
      this.$refs.teamSmallTitleText.classList.add("animate-fadeInDown");
      this.$refs.teamTitleText.classList.add("animate-fadeInDown");

      // Animasi untuk teamPointText (fadeInUp)
      Array.from(teamPointText.children).forEach((element, index) => {
        setTimeout(() => {
          element.classList.add("animate-fadeInUp");
        }, index * 300);
      });
    },
  };
}

function closureAnimation() {
  return {
    init() {
      this.setupIntersectionObserver();
    },
    setupIntersectionObserver() {
      const options = {
        root: null,
        rootMargin: "0px",
        threshold: 0.5,
      };

      const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            this.animateElements();
            observer.disconnect();
          }
        });
      }, options);

      observer.observe(this.$el);
    },
    animateElements() {
      // Animasi untuk Small Title, Title, dan Paragraf (fadeInDown)
      this.$refs.closureSmallTitleText.classList.add("animate-fadeInDown");
      this.$refs.closureTitleText.classList.add("animate-fadeInDown");
      this.$refs.closureParagrafText.classList.add("animate-fadeInDown");

      // Animasi untuk CTA Button (fadeInUp)
      this.$refs.closureCta.classList.add("animate-fadeInUp");
    },
  };
}
