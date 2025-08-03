Chắc chắn rồi! Đổi font là một trong những cách tuyệt vời nhất để cá nhân hóa theme và nâng cao trải nghiệm đọc.
  
Chúng ta sẽ sử dụng một "cặp đôi hoàn hảo" đang rất thịnh hành hiện nay, được cung cấp miễn phí từ Google Fonts:
1.  **Inter:** Một font chữ không chân (sans-serif) siêu sạch sẽ, hiện đại và cực kỳ dễ đọc, lý tưởng cho các đoạn văn bản chính.
2.  **Fira Code:** "Vua" của các font lập trình, nổi tiếng với tính năng "ligatures" (tự động ghép các ký tự như `->` thành →, `==` thành ══), giúp code trông gọn gàng và đẹp mắt hơn rất nhiều.
  
### Cách thực hiện
  
Chúng ta chỉ cần thêm 2 bước vào file CSS của bạn:
1.  **`@import`:** Thêm một dòng ở ngay đầu file CSS để "nhập khẩu" font từ Google Fonts.
2.  **`font-family`:** Cập nhật lại thuộc tính `font-family` ở những nơi cần thiết.
  
Dưới đây là phiên bản nâng cấp của theme **"Modern GitHub Dark"** đã tích hợp sẵn 2 font chữ này. Bạn chỉ cần sao chép toàn bộ và thay thế cho file CSS cũ.
  
---
  
### Theme "Modern GitHub Dark" (Phiên bản Font Nâng cấp)
  
```css
/*
 *  THEME: MODERN GITHUB DARK v2 (FONT UPGRADE)
 *  Sử dụng cặp đôi font chữ Inter (cho văn bản) và Fira Code (cho code).
 *  ---
 *  Dán toàn bộ nội dung này vào file CSS tùy chỉnh của bạn.
*/
  
/* ========================================================================= */
/* BƯỚC 1: NHẬP KHẨU FONT TỪ GOOGLE FONTS                                   */
/* Dòng này phải được đặt ở ngay đầu file.                                 */
/* ========================================================================= */
@import url('https://fonts.googleapis.com/css2?family=Fira+Code&family=Inter:wght@400;700&display=swap');
  
  
/* ------------------------------------------------------------------------- */
/* CÀI ĐẶT TỔNG THỂ (Nền, Font, Màu chữ)                                     */
/* ------------------------------------------------------------------------- */
.markdown-preview.markdown-preview {
  /* Màu nền chính */
  background-color: #0d1117;
  
  /* Màu chữ mặc định */
  color: #c9d1d9;
  
  /* === THAY ĐỔI FONT CHÍNH === */
  /* Sử dụng Inter làm font chính, và các font hệ thống làm phương án dự phòng */
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  
  /* Cỡ chữ và chiều cao dòng dễ đọc */
  font-size: 16px;
  line-height: 1.6;
  
  /* Thêm khoảng đệm xung quanh để ảnh trông đẹp hơn */
  padding: 45px;
}
  
/* ------------------------------------------------------------------------- */
/* TIÊU ĐỀ (h1, h2, h3...)                                                   */
/* ------------------------------------------------------------------------- */
.markdown-preview.markdown-preview h1,
.markdown-preview.markdown-preview h2,
.markdown-preview.markdown-preview h3,
.markdown-preview.markdown-preview h4,
.markdown-preview.markdown-preview h5,
.markdown-preview.markdown-preview h6 {
  color: #58a6ff; /* Màu xanh dương làm điểm nhấn */
  border-bottom-color: #30363d; /* Màu của đường gạch chân */
  /* Làm cho tiêu đề đậm hơn một chút để nổi bật */
  font-weight: 700;
}
  
.markdown-preview.markdown-preview h1 {
  font-size: 2em;
}
  
.markdown-preview.markdown-preview h2 {
  font-size: 1.5em;
}
  
/* ------------------------------------------------------------------------- */
/* LIÊN KẾT (Links)                                                         */
/* ------------------------------------------------------------------------- */
.markdown-preview.markdown-preview a {
  color: #58a6ff;
  text-decoration: none; /* Bỏ gạch chân mặc định */
}
.markdown-preview.markdown-preview a:hover {
  text-decoration: underline; /* Thêm gạch chân khi di chuột qua */
}
  
  
/* ------------------------------------------------------------------------- */
/* KHỐI CODE (Code Blocks)                                                  */
/* ------------------------------------------------------------------------- */
/* Code nằm giữa dòng và nguyên một khối code */
.markdown-preview.markdown-preview code {
  /* === THAY ĐỔI FONT CODE === */
  /* Sử dụng Fira Code với các ligatures tuyệt đẹp */
  font-family: 'Fira Code', 'Consolas', 'Courier New', monospace;
  background-color: rgba(110, 118, 129, 0.2);
  padding: .2em .4em;
  font-size: 85%;
  border-radius: 6px;
}
  
/* Tùy chỉnh riêng cho khối code lớn */
.markdown-preview.markdown-preview pre > code {
  background-color: #161b22;
  border: 1px solid #30363d;
  display: block;
  padding: 16px;
  overflow-x: auto;
}
  
/* ------------------------------------------------------------------------- */
/* CÁC THÀNH PHẦN KHÁC                                                      */
/* ------------------------------------------------------------------------- */
  
/* Đường kẻ ngang */
.markdown-preview.markdown-preview hr {
  border-top: 1px solid #30363d;
  background: transparent;
}
  
/* Trích dẫn */
.markdown-preview.markdown-preview blockquote {
  padding-left: 1em;
  border-left: 0.25em solid #30363d;
  color: #8b949e; /* Màu chữ trích dẫn hơi mờ đi một chút */
}
  
/* Bảng biểu */
.markdown-preview.markdown-preview table {
  border-collapse: collapse;
}
.markdown-preview.markdown-preview th,
.markdown-preview.markdown-preview td {
  border: 1px solid #30363d;
  padding: 8px 12px;
}
.markdown-preview.markdown-preview th {
  background-color: #161b22;
  font-weight: bold;
}
.markdown-preview.markdown-preview tr:nth-child(2n) {
  background-color: #161b22; /* Màu nền cho các hàng chẵn */
}
```
  
### (Tùy chọn nhưng khuyến khích) Cài đặt font vào máy tính
  
Mặc dù `@import` sẽ tự động tải font về, nhưng để có hiệu năng tốt nhất và đảm bảo font hiển thị đúng ngay cả khi không có mạng, bạn nên cài đặt 2 font này trực tiếp vào hệ điều hành của mình:
  
1.  Truy cập Google Fonts: [**Inter**](https://fonts.google.com/specimen/Inter ) và [**Fira Code**](https://fonts.google.com/specimen/Fira+Code ).
2.  Nhấn nút "Download family" trên cả hai trang.
3.  Giải nén file `.zip` và cài đặt các file font (thường là `.ttf` hoặc `.otf`) vào máy tính của bạn (nhấp chuột phải vào file font -> Install).
  
Chỉ với thay đổi nhỏ này, tài liệu của bạn sẽ trông chuyên nghiệp và dễ đọc hơn hẳn.
  