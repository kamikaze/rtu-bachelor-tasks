resource "aws_ecr_repository" "lambda_python_runtime" {
  count = var.ecr_enabled ? 1 : 0
  name  = "lambda-python-runtime"

#  lifecycle {
#    prevent_destroy = true
#  }

  lifecycle {
    ignore_changes = [
      name,
    ]
  }

  tags = {
    Name         = "lambda-python-runtime"
    Organization = "RTU"
  }
}

resource "aws_ecr_repository" "dataset_image_converter" {
  count = var.ecr_enabled ? 1 : 0
  name  = "dataset-image-converter"

  #  lifecycle {
  #    prevent_destroy = true
  #  }

  lifecycle {
    ignore_changes = [
      name,
    ]
  }

  tags = {
    Name         = "dataset-image-converter"
    Organization = "RTU"
  }
}
