/*
This file is part of BGSLibrary.

BGSLibrary is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BGSLibrary is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BGSLibrary.  If not, see <http://www.gnu.org/licenses/>.
*/
#pragma once

#include "IBGS.h"
#include "IMBS/IMBS.hpp"

namespace bgslibrary
{
  namespace algorithms
  {
    class IndependentMultimodal : public IBGS
    {
    private:
      BackgroundSubtractorIMBS* pIMBS;
      int fps;

    public:
      IndependentMultimodal();
      ~IndependentMultimodal();

      void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);

    private:
      void saveConfig();
      void loadConfig();
    };

    static BGS_Register<IndependentMultimodal> register_IndependentMultimodal("IndependentMultimodal");
  }
}