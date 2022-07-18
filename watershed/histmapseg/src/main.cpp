#include <fmt/core.h>

#include <mln/core/image/ndimage.hpp>
#include <mln/core/colors.hpp>
#include <mln/core/image/view/transform.hpp>
#include <mln/core/image/view/operators.hpp>
#include <mln/core/algorithm/transform.hpp>
#include <mln/core/algorithm/accumulate.hpp>

#include <mln/morpho/watershed.hpp>
#include <mln/morpho/area_filter.hpp>
#include <mln/labeling/chamfer_distance_transform.hpp>
#include <mln/io/imread.hpp>
#include <mln/io/imsave.hpp>

#include <mln/core/se/mask2d.hpp>
#include <mln/core/neighborhood/c8.hpp>


const mln::rgb8 regions_table[] = {
    {0, 0, 0},       {255, 255, 255}, {0, 127, 255},   {127, 255, 0},   {255, 0, 127},   {148, 148, 148},
    {0, 255, 128},   {128, 0, 255},   {255, 128, 0},   {1, 0, 159},     {0, 159, 1},     {159, 1, 0},
    {96, 254, 255},  {255, 96, 254},  {254, 255, 96},  {21, 125, 126},  {126, 21, 125},  {125, 126, 21},
    {116, 116, 255}, {116, 255, 116}, {255, 116, 116}, {0, 227, 228},   {228, 0, 227},   {227, 228, 0},
    {28, 27, 255},   {27, 255, 28},   {255, 28, 27},   {59, 59, 59},    {176, 195, 234}, {255, 196, 175},
    {68, 194, 171},  {171, 68, 194},  {194, 171, 68},  {71, 184, 72},   {72, 71, 184},   {184, 72, 71},
    {188, 255, 169}, {63, 179, 252},  {179, 252, 63},  {252, 63, 179},  {0, 9, 80},      {9, 80, 0},
    {80, 0, 9},      {250, 175, 255}, {213, 134, 199}, {95, 100, 115},  {0, 163, 188},   {163, 188, 0},
    {188, 0, 163},   {0, 73, 203},    {73, 203, 0},    {203, 0, 73},    {0, 189, 94},    {94, 0, 189},
    {189, 94, 0},    {119, 243, 187}, {32, 125, 55},   {55, 32, 125},   {125, 55, 32},   {185, 102, 255},
    {255, 185, 102}, {168, 209, 120}, {119, 166, 208}, {192, 96, 135},  {41, 255, 182},  {130, 153, 83},
    {55, 88, 247},   {55, 247, 89},   {247, 55, 88},   {0, 75, 87},     {75, 87, 0},     {87, 0, 75},
    {59, 135, 200},  {127, 213, 51},  {162, 255, 255}, {182, 37, 255},  {255, 182, 37},  {117, 57, 228},
    {210, 163, 142}, {228, 117, 57},  {246, 255, 193}, {123, 107, 188}, {107, 194, 123}, {5, 59, 145},
    {59, 145, 5},    {145, 5, 59},    {198, 39, 119},  {23, 197, 40},   {40, 23, 197},   {197, 40, 23},
    {158, 199, 178}, {121, 201, 255}, {223, 223, 134}, {84, 253, 39},   {15, 203, 149},  {149, 15, 203},
    {203, 149, 15},  {90, 144, 152},  {139, 75, 143},  {132, 97, 71},   {219, 65, 224},  {224, 219, 65},
    {40, 255, 255},  {69, 223, 218},  {0, 241, 74},    {74, 0, 241},    {241, 74, 0},    {51, 171, 122},
    {227, 211, 220}, {87, 127, 61},   {176, 124, 90},  {13, 39, 36},    {255, 142, 165}, {255, 38, 255},
    {255, 255, 38},  {107, 50, 83},   {165, 142, 224}, {9, 181, 255},   {181, 255, 9},   {255, 9, 181},
    {70, 238, 140},  {5, 74, 255},    {255, 5, 74},    {51, 84, 138},   {101, 172, 31},  {17, 115, 177},
    {0, 0, 221},     {0, 221, 0},     {221, 0, 0},     {200, 255, 220}, {50, 41, 0},     {205, 150, 255},
    {116, 45, 178},  {189, 255, 113}, {44, 0, 47},     {171, 119, 40},  {255, 107, 205}, {172, 115, 177},
    {236, 73, 133},  {168, 0, 109},   {207, 46, 168},  {203, 181, 188}, {35, 188, 212},  {52, 97, 90},
    {184, 209, 39},  {152, 164, 41},  {70, 46, 227},   {227, 70, 46},   {255, 156, 211}, {222, 146, 98},
    {95, 56, 136},   {152, 54, 102},  {0, 142, 86},    {86, 0, 142},    {142, 86, 0},    {96, 223, 86},
    {46, 135, 246},  {120, 208, 4},   {158, 233, 212}, {214, 92, 177},  {88, 147, 104},  {147, 240, 149},
    {148, 93, 227},  {133, 255, 72},  {194, 27, 209},  {255, 255, 147}, {0, 93, 44},     {158, 36, 160},
    {0, 233, 182},   {217, 94, 96},   {88, 103, 218},  {38, 154, 163},  {139, 114, 118}, {43, 0, 94},
    {174, 164, 113}, {114, 188, 168}, {119, 23, 0},    {93, 86, 42},    {202, 226, 255}, {155, 191, 80},
    {136, 158, 255}, {62, 247, 0},    {88, 146, 234},  {229, 183, 0},   {36, 212, 110},  {161, 143, 0},
    {210, 191, 105}, {0, 164, 133},   {89, 30, 41},    {132, 0, 164},   {42, 89, 30},    {217, 222, 178},
    {11, 22, 121},   {22, 107, 221},  {255, 151, 69},  {3, 158, 45},    {45, 3, 158},    {158, 45, 3},
    {29, 42, 86},    {22, 122, 9},    {110, 209, 213}, {57, 221, 53},   {91, 101, 159},  {45, 140, 93},
    {37, 213, 247},  {0, 34, 185},    {34, 185, 0},    {185, 0, 34},    {172, 0, 236},   {78, 180, 210},
    {221, 107, 231}, {43, 49, 162},   {49, 162, 43},   {162, 43, 49},   {213, 248, 36},  {214, 0, 114},
    {248, 36, 213},  {243, 34, 149},  {167, 158, 185}, {224, 122, 144}, {149, 245, 34},  {98, 31, 255},
    {255, 98, 31},   {193, 200, 152}, {95, 80, 255},   {63, 123, 128},  {72, 62, 102},   {148, 62, 255},
    {108, 226, 151}, {255, 99, 159},  {126, 255, 226}, {136, 223, 98},  {255, 95, 80},   {15, 153, 225},
    {211, 41, 73},   {41, 71, 212},   {187, 217, 83},  {79, 235, 180},  {127, 166, 0},   {243, 135, 251},
    {0, 41, 229},    {229, 0, 41},    {216, 255, 82},  {249, 174, 141}, {255, 215, 249}, {79, 31, 167},
    {167, 79, 31},   {185, 102, 213}, {83, 215, 255},  {40, 2, 4},      {220, 171, 224}, {4, 0, 41},
    {90, 50, 6},     {113, 15, 221},  {221, 113, 15},  {115, 0, 33},
};

const std::size_t regions_table_size = sizeof(regions_table) / sizeof(mln::rgb8);



struct myvisitor
{
  using P = mln::point2d;

  void on_make_set(P p) noexcept {
    m_attribute(p) = { 1, m_ima(p) };
  }

  P on_union(mln::dontcare_t, P rp, mln::dontcare_t, P rq) noexcept {
    auto &a = m_attribute(rp);
    auto &b = m_attribute(rq);
    a.count += b.count;
    a.min = std::min(a.min, b.min);
    return rp;
  }

  void on_finish(P p, P root) noexcept { m_ima(p) = m_ima(root); }

  bool test(P p) const noexcept {
    auto a = m_attribute(p);
    return a.count >= m_area && (m_ima(p) - a.min) >= m_dynamic;
  }

  myvisitor(mln::image2d<uint8_t> &input, int area, int dynamic)
      : m_ima(input), m_attribute(input.domain()),
        m_area(area), m_dynamic{dynamic} //
  {
  }

private:
  struct attr_t {
    int count;
    int min;
  };

  mln::image2d<uint8_t> m_ima;
  mln::image2d<attr_t> m_attribute;
  int m_area;
  int m_dynamic;
};





mln::image2d<uint8_t> convert_to_u8(mln::ndbuffer_image input)
{
  if (auto* a = input.cast_to<uint8_t, 2>(); a)
    return *a;
  else if (auto* a = input.cast_to<mln::rgb8, 2>(); a)
    return mln::transform(*a, [](mln::rgb8 x) -> uint8_t { return x[0]; });

  throw std::runtime_error("Invalid input format (expected RGB8 or UINT8 2D image)");
}


int main(int argc, char** argv)
{
  if (argc < 6)
  {
    fmt::print(stderr, "Usage: {} input.png dynamic area_closing ws.tiff out.png\n", argv[0]);
    std::abort();
  }


  const char* ipath = argv[1];
  const char* opath1 = argv[4];
  const char* opath2 = argv[5];
  //const int threshold = std::atoi(argv[2]);
  const int dynamic = std::atoi(argv[2]);
  const int area = std::atoi(argv[3]);

  mln::ndbuffer_image original = mln::io::imread(ipath);
  mln::image2d<uint8_t> input = convert_to_u8(original);

  // Get the r component + threshold
  // mln::image2d<bool> binary_input = mln::transform(
  //     input_rgb, [threshold](mln::rgb8 v) -> bool { return v[0] < threshold; });

  // Distance transform
  // auto weights = mln::se::wmask2d({{+3, +2, +3}, {+2, +0, +2}, {+3, +2, +3}});
  // auto distance = mln::labeling::chamfer_distance_transform<uint16_t>(
  //   binary_input, weights, true);



  // Inverse the distance
  // uint16_t m = mln::accumulate(distance, uint16_t(0), [](uint16_t a, uint16_t b) { return std::max(a, b); });
  // auto dinv = mln::view::transform(distance, [m](uint16_t x) -> uint16_t { return m - x; });
  // mln::io::imsave(dinv, "distance.tiff");

  // Make a closing
  myvisitor viz(input, area, dynamic);
  mln::morpho::canvas::union_find(input, mln::c8, viz, std::greater<uint8_t>());
  //auto closed = mln::morpho::area_closing(input, mln::c8, area);
  //mln::io::imsave(input, "closed.tiff");

// (3) Run the watershed segmentation
  int  nlabel;
  auto ws = mln::morpho::watershed<int16_t>(input, mln::c8, nlabel);

  // Output
  mln::io::imsave(ws, opath1);


  auto res = mln::transform(ws, [](int label) { return regions_table[label % regions_table_size]; });
  mln::io::imsave(res, opath2);
}
